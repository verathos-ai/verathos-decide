//! Lowercase hex encoding and decoding.

const DIGITS: &[u8; 16] = b"0123456789abcdef";

pub fn push_encoded(out: &mut String, bytes: &[u8]) {
    out.reserve(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
}

pub fn encode(bytes: &[u8]) -> String {
    let mut out = String::new();
    push_encoded(&mut out, bytes);
    out
}

/// Decodes an even-length hex string in either case.
pub fn decode(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 {
        return None;
    }
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| Some(digit(pair[0])? << 4 | digit(pair[1])?))
        .collect()
}

fn digit(symbol: u8) -> Option<u8> {
    match symbol {
        b'0'..=b'9' => Some(symbol - b'0'),
        b'a'..=b'f' => Some(symbol - b'a' + 10),
        b'A'..=b'F' => Some(symbol - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_rejects_malformed_input() {
        let bytes = [0x00, 0x7f, 0x80, 0xab, 0xff];
        assert_eq!(encode(&bytes), "007f80abff");
        assert_eq!(decode("007f80abff").unwrap(), bytes);
        assert_eq!(decode("007F80ABFF").unwrap(), bytes);
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
        assert_eq!(decode("abc"), None);
        assert_eq!(decode("0g"), None);
    }
}
