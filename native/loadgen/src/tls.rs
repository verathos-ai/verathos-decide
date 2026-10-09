//! TLS to the endpoints: each target's certificate is accepted by its pin, the SHA-256 of its
//! SubjectPublicKeyInfo, which the plan gives per target. Endpoints are reached by IP address with
//! self-signed certificates, so no CA, name or validity check applies; the handshake signature is
//! still verified against the pinned key, so only the key's holder completes a handshake.

use rustls::{
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider},
    pki_types::{CertificateDer, ServerName, UnixTime},
    CertificateError, ClientConfig, DigitallySignedStruct, Error, SignatureScheme,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub type Pin = [u8; 32];

/// A client configuration that accepts exactly the certificate whose pin is `pin`.
pub fn client_config(pin: Pin) -> ClientConfig {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("the ring provider supports the default protocol versions")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(PinVerifier { pin, provider }))
        .with_no_client_auth()
}

/// The pin of a DER certificate, or `None` if it is not one.
pub fn pin_of(certificate: &[u8]) -> Option<Pin> {
    subject_public_key_info(certificate).map(|spki| Sha256::digest(spki).into())
}

#[derive(Debug)]
struct PinVerifier {
    pin: Pin,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for PinVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        match pin_of(end_entity) {
            Some(pin) if pin == self.pin => Ok(ServerCertVerified::assertion()),
            Some(_) => Err(Error::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure,
            )),
            None => Err(Error::InvalidCertificate(CertificateError::BadEncoding)),
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// The DER of a certificate's subjectPublicKeyInfo: `Certificate ::= SEQUENCE { tbsCertificate
/// SEQUENCE { [0] version OPTIONAL, serialNumber, signature, issuer, validity, subject,
/// subjectPublicKeyInfo, ... }, ... }`.
fn subject_public_key_info(certificate: &[u8]) -> Option<&[u8]> {
    let (certificate, rest) = element(certificate, SEQUENCE)?;
    if !rest.is_empty() {
        return None;
    }
    let (tbs, _) = element(content(certificate)?, SEQUENCE)?;
    let mut fields = content(tbs)?;
    if fields.first() == Some(&VERSION) {
        fields = element(fields, VERSION)?.1;
    }
    for tag in [INTEGER, SEQUENCE, SEQUENCE, SEQUENCE, SEQUENCE] {
        fields = element(fields, tag)?.1;
    }
    Some(element(fields, SEQUENCE)?.0)
}

const INTEGER: u8 = 0x02;
const SEQUENCE: u8 = 0x30;
const VERSION: u8 = 0xa0;

/// The DER element at the start of `input` with tag `tag` (its tag, length and content), and what
/// follows it.
fn element(input: &[u8], tag: u8) -> Option<(&[u8], &[u8])> {
    if *input.first()? != tag {
        return None;
    }
    let (header, length) = match *input.get(1)? {
        short @ 0..=0x7f => (2, usize::from(short)),
        long @ 0x81..=0x84 => {
            let octets = usize::from(long & 0x7f);
            let bytes = input.get(2..2 + octets)?;
            if bytes[0] == 0 {
                return None; // not the shortest form
            }
            let length = bytes
                .iter()
                .fold(0usize, |acc, byte| acc << 8 | usize::from(*byte));
            if length < 0x80 {
                return None;
            }
            (2 + octets, length)
        }
        _ => return None,
    };
    let end = header.checked_add(length)?;
    Some((input.get(..end)?, &input[end..]))
}

/// The content of a whole DER element.
fn content(element: &[u8]) -> Option<&[u8]> {
    let header = match *element.get(1)? {
        0..=0x7f => 2,
        long => 2 + usize::from(long & 0x7f),
    };
    element.get(header..)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex;

    /// `tests/fixtures/server.crt.der`, an ECDSA P-256 certificate, and its pin.
    const CERTIFICATE: &[u8] = include_bytes!("../tests/fixtures/server.crt.der");
    const PIN: &str = include_str!("../tests/fixtures/server.pin");

    #[test]
    fn finds_the_pin_of_a_certificate() {
        assert_eq!(
            hex::encode(&pin_of(CERTIFICATE).unwrap()),
            PIN.trim(),
            "the pin is the SHA-256 of the subjectPublicKeyInfo"
        );
    }

    #[test]
    fn refuses_what_is_not_a_certificate() {
        assert_eq!(pin_of(b""), None);
        assert_eq!(pin_of(&CERTIFICATE[..CERTIFICATE.len() - 1]), None);
        assert_eq!(pin_of(&[CERTIFICATE, b"\0"].concat()), None);
        assert_eq!(pin_of(&[0x30, 0x80]), None);
        assert_eq!(pin_of(&[0x30, 0x81, 0x05, 0, 0, 0, 0, 0]), None);
        assert_eq!(pin_of(&[0x30, 0x84, 0xff, 0xff, 0xff, 0xff]), None);
    }
}
