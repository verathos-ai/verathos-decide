//! The images of a request with images, drawn from a 32-byte seed and encoded as canonical PNG,
//! byte for byte as `verathos.throughput.images` draws and encodes them.
//!
//! The seed is stretched by SHA-256 in counter mode. The request's images share one background
//! colour; each image has its width and height drawn from the mix's sides and one to three
//! rectangles in other colours, each up to 3/8 of the image's width and height. The PNG is 8-bit
//! RGB, not interlaced, its image data right after its header: one deflate block with the fixed
//! Huffman codes, where a row that repeats the row above is filtered with Up (all zeros) and any
//! other row is unfiltered and coded as runs of one colour (each run's first pixel as literals,
//! the rest as copies three bytes back).

use sha2::{Digest, Sha256};
use std::ops::RangeInclusive;

pub const SEED_BYTES: usize = 32;
/// White, black, blue, yellow, grey and green (a request's background is one of these), red and
/// orange.
const COLOURS: [[u8; 3]; 8] = [
    [250, 250, 250],
    [20, 20, 20],
    [40, 90, 220],
    [240, 210, 40],
    [150, 150, 150],
    [40, 170, 70],
    [220, 40, 40],
    [245, 140, 30],
];
const BACKGROUNDS: u64 = 6;
const MAX_RECTANGLES: u64 = 3;
const PNG_START: [u8; 8] = *b"\x89PNG\r\n\x1a\n";
/// Deflate with a 32 KiB window.
const ZLIB_HEADER: [u8; 2] = [0x78, 0x01];
const MAX_COPY: u32 = 258;
const LENGTH_BASE: [u32; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const ADLER_MODULUS: u64 = 65_521;
const CRC_TABLE: [u32; 256] = crc_table();

struct Rectangle {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    colour: usize,
}

/// Bytes drawn from a seed: SHA-256 of the seed and a 64-bit big-endian block counter, block
/// after block.
struct Stream {
    seed: [u8; SEED_BYTES],
    counter: u64,
    block: [u8; 32],
    used: usize,
}

impl Stream {
    fn new(seed: [u8; SEED_BYTES]) -> Self {
        Self {
            seed,
            counter: 0,
            block: [0; 32],
            used: 32,
        }
    }

    fn byte(&mut self) -> u8 {
        if self.used == self.block.len() {
            let mut hasher = Sha256::new();
            hasher.update(self.seed);
            hasher.update(self.counter.to_be_bytes());
            self.block = hasher.finalize().into();
            self.counter += 1;
            self.used = 0;
        }
        self.used += 1;
        self.block[self.used - 1]
    }

    /// A number in [0, `n`): the next 8 bytes as a big-endian number, times `n`, over 2^64.
    fn below(&mut self, n: u64) -> u64 {
        let value = (0..8).fold(0u64, |value, _| value << 8 | u64::from(self.byte()));
        ((u128::from(value) * u128::from(n)) >> 64) as u64
    }

    /// An index into a list of `len` items.
    fn index(&mut self, len: usize) -> usize {
        self.below(len as u64) as usize
    }

    /// A number in [0, `n`].
    fn up_to(&mut self, n: u32) -> u32 {
        self.below(u64::from(n) + 1) as u32
    }
}

/// The PNGs of a request's images: as many as `count` gives, each side from `sides`.
pub fn draw(seed: [u8; SEED_BYTES], count: &RangeInclusive<u32>, sides: &[u32]) -> Vec<Vec<u8>> {
    let mut stream = Stream::new(seed);
    let images = count.start() + stream.up_to(count.end() - count.start());
    let background = stream.below(BACKGROUNDS) as usize;
    (0..images)
        .map(|_| {
            let width = sides[stream.index(sides.len())];
            let height = sides[stream.index(sides.len())];
            let rectangles = (0..=stream.below(MAX_RECTANGLES))
                .map(|_| {
                    let w = (width / 8).max(1) + stream.up_to(width / 4);
                    let h = (height / 8).max(1) + stream.up_to(height / 4);
                    let x = stream.up_to(width - w);
                    let y = stream.up_to(height - h);
                    let colour = stream.index(COLOURS.len() - 1);
                    Rectangle {
                        x,
                        y,
                        width: w,
                        height: h,
                        colour: colour + usize::from(colour >= background),
                    }
                })
                .collect::<Vec<_>>();
            png(width, height, background, &rectangles)
        })
        .collect()
}

/// A row's pixels as (colour, length) runs, each as long as it goes: the background under the
/// rectangles that cross the row, the later ones on top.
fn runs(width: u32, row: u32, background: usize, rectangles: &[Rectangle]) -> Vec<(usize, u32)> {
    let crossing = rectangles
        .iter()
        .filter(|r| r.y <= row && row < r.y + r.height)
        .collect::<Vec<_>>();
    let mut edges = vec![0, width];
    edges.extend(crossing.iter().flat_map(|r| [r.x, r.x + r.width]));
    edges.sort_unstable();
    edges.dedup();
    let mut out: Vec<(usize, u32)> = Vec::new();
    for pair in edges.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        let colour = crossing
            .iter()
            .rev()
            .find(|r| r.x <= start && start < r.x + r.width)
            .map_or(background, |r| r.colour);
        match out.last_mut() {
            Some((last, length)) if *last == colour => *length += end - start,
            _ => out.push((colour, end - start)),
        }
    }
    out
}

/// The canonical PNG of a background colour with `rectangles` on it.
fn png(width: u32, height: u32, background: usize, rectangles: &[Rectangle]) -> Vec<u8> {
    let mut bits = Bits::default();
    let mut checksum = Adler::default();
    bits.put(0b011, 3); // the last block, with the fixed Huffman codes
    let mut above: Option<Vec<(usize, u32)>> = None;
    for y in 0..height {
        let changes = y == 0 || rectangles.iter().any(|r| y == r.y || y == r.y + r.height);
        if changes {
            let row = runs(width, y, background, rectangles);
            if above.as_ref() != Some(&row) {
                bits.unfiltered(&row);
                checksum.update(&[0]);
                for &(colour, length) in &row {
                    for _ in 0..length {
                        checksum.update(&COLOURS[colour]);
                    }
                }
                above = Some(row);
                continue;
            }
        }
        bits.up(width);
        checksum.update(&[2]);
        checksum.zeros(3 * u64::from(width));
    }
    bits.put(0, 7); // the end of the block
    let mut data = ZLIB_HEADER.to_vec();
    data.extend(bits.done());
    data.extend(checksum.value().to_be_bytes());
    let mut header = Vec::with_capacity(13);
    header.extend(width.to_be_bytes());
    header.extend(height.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]);
    let mut out = PNG_START.to_vec();
    chunk(&mut out, b"IHDR", &header);
    chunk(&mut out, b"IDAT", &data);
    chunk(&mut out, b"IEND", &[]);
    out
}

/// The lengths `length` (at least 3) bytes are copied in: 258 each, but the last two shortened as
/// far as needed to keep each at least 3.
fn copies(mut length: u32) -> impl Iterator<Item = u32> {
    std::iter::from_fn(move || {
        if length == 0 {
            return None;
        }
        let mut step = length.min(MAX_COPY);
        if (1..3).contains(&(length - step)) {
            step = length - 3;
        }
        length -= step;
        Some(step)
    })
}

/// Bits written into bytes from the least significant bit up, as deflate writes them.
#[derive(Default)]
struct Bits {
    out: Vec<u8>,
    value: u64,
    count: u32,
}

impl Bits {
    fn put(&mut self, value: u32, count: u32) {
        self.value |= u64::from(value) << self.count;
        self.count += count;
        while self.count >= 8 {
            self.out.push(self.value as u8);
            self.value >>= 8;
            self.count -= 8;
        }
    }

    /// A code of the fixed Huffman codes, which deflate writes from its most significant bit.
    fn code(&mut self, code: u32, count: u32) {
        self.put(code.reverse_bits() >> (32 - count), count);
    }

    fn literal(&mut self, value: u8) {
        match value {
            0..=143 => self.code(0x30 + u32::from(value), 8),
            _ => self.code(0x190 + u32::from(value) - 144, 9),
        }
    }

    /// A copy of `length` (3 to 258) bytes from `distance` (1 to 4) bytes back.
    fn copy(&mut self, length: u32, distance: u32) {
        let index = LENGTH_BASE
            .iter()
            .rposition(|&base| base <= length)
            .unwrap_or(0);
        let symbol = 257 + index as u32;
        if symbol < 280 {
            self.code(symbol - 256, 7);
        } else {
            self.code(0xC0 + symbol - 280, 8);
        }
        self.put(length - LENGTH_BASE[index], LENGTH_EXTRA[index]);
        self.code(distance - 1, 5);
    }

    fn unfiltered(&mut self, row: &[(usize, u32)]) {
        self.literal(0);
        for &(colour, length) in row {
            COLOURS[colour]
                .iter()
                .for_each(|&value| self.literal(value));
            if length > 1 {
                copies(3 * (length - 1)).for_each(|step| self.copy(step, 3));
            }
        }
    }

    /// A row that repeats the one above, filtered with Up: the filter type and zeros.
    fn up(&mut self, width: u32) {
        self.literal(2);
        self.literal(0);
        let zeros = 3 * width - 1;
        if zeros >= 3 {
            copies(zeros).for_each(|step| self.copy(step, 1));
        } else {
            (0..zeros).for_each(|_| self.literal(0));
        }
    }

    fn done(mut self) -> Vec<u8> {
        if self.count > 0 {
            self.out.push(self.value as u8);
        }
        self.out
    }
}

/// zlib's checksum of the image data.
struct Adler {
    a: u64,
    b: u64,
}

impl Default for Adler {
    fn default() -> Self {
        Self { a: 1, b: 0 }
    }
}

impl Adler {
    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.a = (self.a + u64::from(byte)) % ADLER_MODULUS;
            self.b = (self.b + self.a) % ADLER_MODULUS;
        }
    }

    /// `count` zero bytes: each adds the running sum to the second sum.
    fn zeros(&mut self, count: u64) {
        self.b = (self.b + count % ADLER_MODULUS * self.a) % ADLER_MODULUS;
    }

    fn value(&self) -> u32 {
        (self.b << 16 | self.a) as u32
    }
}

const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 1 {
                0xEDB8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

/// A PNG chunk: its length, type, data and the CRC-32 of its type and data.
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend((data.len() as u32).to_be_bytes());
    out.extend(kind);
    out.extend(data);
    let crc = kind.iter().chain(data).fold(!0u32, |crc, &byte| {
        CRC_TABLE[((crc ^ u32::from(byte)) & 0xFF) as usize] ^ (crc >> 8)
    });
    out.extend((!crc).to_be_bytes());
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with its padding.
pub fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(base64_len(bytes.len()));
    for group in bytes.chunks(3) {
        let value = group.iter().enumerate().fold(0u32, |value, (i, &byte)| {
            value | u32::from(byte) << (16 - 8 * i)
        });
        for i in 0..4 {
            out.push(if i <= group.len() {
                BASE64[(value >> (18 - 6 * i) & 0x3F) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

pub fn base64_len(bytes: usize) -> usize {
    bytes.div_ceil(3) * 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex;
    use serde_json::Value;

    fn size(png: &[u8]) -> (u32, u32) {
        let field = |at: usize| u32::from_be_bytes(png[at..at + 4].try_into().unwrap());
        (field(16), field(20))
    }

    /// The vectors `verathos.throughput.images` draws (`tests/fixtures/images.json`): the same
    /// seeds give the same bytes in both languages.
    #[test]
    fn draws_the_bytes_the_python_code_draws() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/images.json")).unwrap();
        for vector in fixture["vectors"].as_array().unwrap() {
            let seed = hex::decode(vector["seed"].as_str().unwrap()).unwrap();
            let range = |key: &str, index: usize| vector[key][index].as_u64().unwrap() as u32;
            let sides = vector["sides_px"]
                .as_array()
                .unwrap()
                .iter()
                .map(|side| side.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            let drawn = draw(
                seed.try_into().unwrap(),
                &(range("count", 0)..=range("count", 1)),
                &sides,
            );
            let expected = vector["images"].as_array().unwrap();
            assert_eq!(drawn.len(), expected.len());
            for (png, image) in drawn.iter().zip(expected) {
                let field = |key: &str| image[key].as_u64().unwrap();
                assert_eq!(size(png), (field("width") as u32, field("height") as u32));
                assert_eq!(png.len() as u64, field("bytes"));
                assert_eq!(
                    hex::encode(&Sha256::digest(png)),
                    image["sha256"].as_str().unwrap()
                );
            }
        }
    }

    #[test]
    fn a_png_has_its_header_then_its_image_data_and_valid_checksums() {
        let sides = [256, 288, 320, 352, 384];
        for index in 0..50u8 {
            for png in draw([index; SEED_BYTES], &(1..=2), &sides) {
                assert_eq!(png[..8], PNG_START);
                let (width, height) = size(&png);
                assert!(sides.contains(&width) && sides.contains(&height));
                assert_eq!(png[24..29], [8, 2, 0, 0, 0]);
                assert_eq!(&png[37..41], b"IDAT");
                let mut at = 8;
                let mut kinds = Vec::new();
                while at < png.len() {
                    let length = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
                    let (kind, data) = (&png[at + 4..at + 8], &png[at + 8..at + 8 + length]);
                    let mut again = Vec::new();
                    chunk(&mut again, kind.try_into().unwrap(), data);
                    assert_eq!(again, png[at..at + 12 + length]);
                    kinds.push(kind.to_vec());
                    at += 12 + length;
                }
                assert_eq!(
                    kinds,
                    [b"IHDR".to_vec(), b"IDAT".to_vec(), b"IEND".to_vec()]
                );
            }
        }
    }

    #[test]
    fn checksums_are_zlibs_and_pngs() {
        let mut adler = Adler::default();
        adler.update(b"Wikipedia");
        assert_eq!(adler.value(), 0x11E6_0398);
        let mut zeros = Adler::default();
        zeros.update(&[7]);
        zeros.zeros(100_000);
        let mut slow = Adler::default();
        slow.update(&[7]);
        slow.update(&vec![0; 100_000]);
        assert_eq!(zeros.value(), slow.value());
        let mut out = Vec::new();
        chunk(&mut out, b"IEND", &[]);
        assert_eq!(out[8..], [0xAE, 0x42, 0x60, 0x82]);
    }

    #[test]
    fn requests_are_reproducible_and_unique() {
        let sides = [256, 512];
        let first = draw([1; SEED_BYTES], &(4..=4), &sides);
        let again = draw([1; SEED_BYTES], &(4..=4), &sides);
        let other = draw([2; SEED_BYTES], &(4..=4), &sides);
        assert_eq!(first, again);
        assert!(first.iter().all(|png| !other.contains(png)));
        assert_eq!(first.len(), 4);
    }

    #[test]
    fn copies_split_lengths_into_valid_steps() {
        for length in 3..2_000 {
            let steps = copies(length).collect::<Vec<_>>();
            assert_eq!(steps.iter().sum::<u32>(), length);
            assert!(steps.iter().all(|step| (3..=MAX_COPY).contains(step)));
        }
        assert_eq!(copies(260).collect::<Vec<_>>(), [257, 3]);
    }

    #[test]
    fn base64_is_standard_with_padding() {
        for (bytes, text) in [
            (&b""[..], ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foobar", "Zm9vYmFy"),
            (&[0xFB, 0xFF, 0xBF], "+/+/"),
        ] {
            assert_eq!(base64(bytes), text);
            assert_eq!(base64_len(bytes.len()), text.len());
        }
    }
}
