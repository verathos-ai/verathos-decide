//! Synthetic requests.
//!
//! Request `k` of target `i` draws all its randomness from its own ChaCha stream, keyed by the plan
//! seed and numbered by `(i, k)`. A plan therefore reproduces its requests exactly, requests can be
//! built in any order on any thread, and no two requests of a run share randomness. For a target
//! whose mix has images, the stream then draws whether the request carries images and, if it does,
//! the seed of its images (`images`) and the questions of a request with images.

use crate::{
    hex, images,
    plan::{Target, MAX_RATE_RPS, MAX_RUN_S, MAX_TARGETS},
};
use bytes::Bytes;
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::Serialize;
use serde_json::value::RawValue;

pub const REQUEST_ID_BYTES: usize = 16;
pub const TICKET_BYTES: usize = 8;
/// Length of `{"model":"","state":"","questions":}`.
const BODY_OVERHEAD: usize = 36;
const IMAGES_OVERHEAD: usize = r#","images":[]"#.len();
const IMAGE_OVERHEAD: usize = r#"{"content_type":"image/png","base64":""}"#.len();
/// Low bits of a request's stream number: its position in the target's schedule. The bits above
/// hold the target index.
const POSITION_BITS: u32 = 40;
/// Stream numbers with the top bit set hold per-target randomness, never a request.
const TARGET_STREAM: u64 = 1 << 63;

const _: () = assert!((MAX_RATE_RPS as u64) * (MAX_RUN_S as u64) < 1 << POSITION_BITS);
const _: () = assert!(((MAX_TARGETS as u64) << POSITION_BITS) < TARGET_STREAM);

pub struct Generator {
    key: [u8; 32],
    words: Vec<String>,
}

pub struct GeneratedRequest {
    /// `REQUEST_ID_BYTES` random bytes as lowercase hex.
    pub id: String,
    /// The serialized JSON body.
    pub body: Bytes,
}

#[derive(Serialize)]
struct Body<'a> {
    model: &'a str,
    state: &'a str,
    questions: &'a RawValue,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    images: Vec<Image>,
}

#[derive(Serialize)]
struct Image {
    content_type: &'static str,
    base64: String,
}

impl Generator {
    pub fn new(key: [u8; 32], words: Vec<String>) -> Self {
        Self { key, words }
    }

    pub fn request(&self, target_index: usize, position: u64, target: &Target) -> GeneratedRequest {
        let mut rng = self.stream(((target_index as u64) << POSITION_BITS) | position);
        let mut id = [0u8; REQUEST_ID_BYTES];
        rng.fill_bytes(&mut id);
        let mut ticket = [0u8; TICKET_BYTES];
        rng.fill_bytes(&mut ticket);
        let count = rng.random_range(target.state_words.clone());
        let mut state = String::with_capacity(32 + count as usize * 9);
        state.push_str("Ticket ");
        hex::push_encoded(&mut state, &ticket);
        state.push_str(". ");
        for index in 0..count {
            if index > 0 {
                state.push(' ');
            }
            state.push_str(&self.words[rng.random_range(0..self.words.len())]);
        }
        state.push('.');
        let (questions, items) = match &target.images {
            Some(mix) if rng.random::<f64>() < mix.share => {
                let mut seed = [0u8; images::SEED_BYTES];
                rng.fill_bytes(&mut seed);
                let items = images::draw(seed, &mix.count, &mix.sides)
                    .iter()
                    .map(|png| Image {
                        content_type: "image/png",
                        base64: images::base64(png),
                    })
                    .collect();
                (&*mix.questions, items)
            }
            _ => (&*target.questions, Vec::new()),
        };
        let images_len = match items.len() {
            0 => 0,
            count => {
                IMAGES_OVERHEAD + count - 1
                    + items
                        .iter()
                        .map(|image| IMAGE_OVERHEAD + image.base64.len())
                        .sum::<usize>()
            }
        };
        // Sized for text without escapes, which base64 never holds: a kept body holds no spare
        // capacity.
        let mut body = Vec::with_capacity(
            BODY_OVERHEAD + target.model.len() + state.len() + questions.get().len() + images_len,
        );
        let fields = Body {
            model: &target.model,
            state: &state,
            questions,
            images: items,
        };
        serde_json::to_writer(&mut body, &fields)
            .expect("string fields and a raw JSON value always serialize");
        GeneratedRequest {
            id: hex::encode(&id),
            body: body.into(),
        }
    }

    /// Randomness of one target's schedule phase and sample choice.
    pub fn target_rng(&self, target_index: usize) -> ChaCha12Rng {
        self.stream(TARGET_STREAM | target_index as u64)
    }

    fn stream(&self, stream: u64) -> ChaCha12Rng {
        let mut rng = ChaCha12Rng::from_seed(self.key);
        rng.set_stream(stream);
        rng
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{tests::plan_json, Plan};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::HashSet;

    fn plan(state_words: (u32, u32)) -> Plan {
        let mut plan = plan_json();
        plan["targets"][0]["state_words"] = json!(state_words);
        let mut second = plan["targets"][0].clone();
        second["id"] = json!("hotkey/gpu1");
        plan["targets"].as_array_mut().unwrap().push(second);
        Plan::parse(plan.to_string().as_bytes(), crate::plan::tests::NOW_MS).unwrap()
    }

    fn generator(plan: &Plan) -> Generator {
        Generator::new(plan.key, plan.words.clone())
    }

    /// A plan whose second target's mix has images.
    fn plan_with_images(share: f64) -> Plan {
        let mut plan = plan_json();
        let mut second = plan["targets"][0].clone();
        second["id"] = json!("hotkey/gpu1");
        second["images"] = json!({
            "share": share,
            "count": [1, 2],
            "sides_px": [32, 64],
            "questions": {"intent": {"type": "noul"}, "background": {"type": "noul"}},
        });
        plan["targets"].as_array_mut().unwrap().push(second);
        Plan::parse(plan.to_string().as_bytes(), crate::plan::tests::NOW_MS).unwrap()
    }

    #[test]
    fn body_has_the_request_shape() {
        let plan = plan((3, 7));
        let generator = generator(&plan);
        let target = &plan.targets[0];
        let mut lengths = HashSet::new();
        for position in 0..500 {
            let request = generator.request(0, position, target);
            assert_eq!(request.id.len(), 2 * REQUEST_ID_BYTES);
            assert!(request
                .id
                .bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
            let text = std::str::from_utf8(&request.body).unwrap();
            assert!(
                text.starts_with(r#"{"model":"kev-4b","state":"Ticket "#),
                "{text}"
            );
            assert!(text.ends_with(&format!(r#","questions":{}}}"#, target.questions.get())));
            let body: Value = serde_json::from_str(text).unwrap();
            assert_eq!(body.as_object().unwrap().len(), 3);
            let state = body["state"].as_str().unwrap();
            let rest = state.strip_prefix("Ticket ").unwrap();
            let (ticket, words) = rest.split_at(2 * TICKET_BYTES);
            assert!(ticket.bytes().all(|c| c.is_ascii_hexdigit()));
            let words = words.strip_prefix(". ").unwrap().strip_suffix('.').unwrap();
            let words = words.split(' ').collect::<Vec<_>>();
            assert!((3..=7).contains(&words.len()));
            assert!(words
                .iter()
                .all(|word| plan.words.iter().any(|known| known == word)));
            lengths.insert(words.len());
        }
        assert_eq!(
            lengths.len(),
            5,
            "every word count of the range should occur"
        );
    }

    #[test]
    fn requests_are_unique_across_positions_and_targets() {
        let plan = plan((1, 1));
        let generator = generator(&plan);
        let mut ids = HashSet::new();
        let mut tickets = HashSet::new();
        for (index, target) in plan.targets.iter().enumerate() {
            for position in 0..20_000 {
                let request = generator.request(index, position, target);
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                assert!(ids.insert(request.id));
                assert!(tickets.insert(body["state"].as_str().unwrap()[7..23].to_owned()));
            }
        }
    }

    /// The requests of a mix without images, as this plan gives them, pinned by their digest.
    #[test]
    fn text_requests_keep_their_bytes() {
        let plan = Plan::parse(
            plan_json().to_string().as_bytes(),
            crate::plan::tests::NOW_MS,
        )
        .unwrap();
        let generator = generator(&plan);
        let mut digest = Sha256::new();
        for position in 0..1000 {
            let request = generator.request(0, position, &plan.targets[0]);
            digest.update(request.id.as_bytes());
            digest.update(&request.body);
        }
        assert_eq!(
            hex::encode(&digest.finalize()),
            "36264aa34bb2d630b9b674a92f3b8bfae861e61210bf6714825cd96941c2c2c5"
        );
    }

    #[test]
    fn a_share_of_requests_carries_images_and_asks_about_them() {
        let plan = plan_with_images(0.25);
        let generator = generator(&plan);
        let (text, mix) = (&plan.targets[0], &plan.targets[1]);
        let mut pngs = HashSet::new();
        let mut with_images = 0;
        for position in 0..4000 {
            let request = generator.request(1, position, mix);
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let plain: Value =
                serde_json::from_slice(&generator.request(1, position, text).body).unwrap();
            // the text is drawn first, so it is the text a target without images gets
            assert_eq!(body["state"], plain["state"]);
            let Some(images) = body.get("images") else {
                assert_eq!(body["questions"], plain["questions"]);
                assert_eq!(body.as_object().unwrap().len(), 3);
                continue;
            };
            with_images += 1;
            assert_eq!(body.as_object().unwrap().len(), 4);
            let questions = mix.images.as_ref().unwrap().questions.get();
            assert_eq!(
                body["questions"],
                serde_json::from_str::<Value>(questions).unwrap()
            );
            let images = images.as_array().unwrap();
            assert!((1..=2).contains(&images.len()));
            for image in images {
                assert_eq!(image.as_object().unwrap().len(), 2);
                assert_eq!(image["content_type"], "image/png");
                let encoded = image["base64"].as_str().unwrap();
                assert!(encoded.starts_with("iVBORw0KGgo"), "{encoded}"); // the PNG signature
                assert!(pngs.insert(encoded.to_owned()), "an image twice");
            }
        }
        assert!(
            (850..1150).contains(&with_images),
            "{with_images} of 4000 with images"
        );
    }

    #[test]
    fn every_request_carries_images_at_a_share_of_one() {
        let plan = plan_with_images(1.0);
        let generator = generator(&plan);
        assert!((0..200).all(|position| {
            let body: Value =
                serde_json::from_slice(&generator.request(1, position, &plan.targets[1]).body)
                    .unwrap();
            body.get("images").is_some()
        }));
    }

    #[test]
    fn requests_are_reproducible_from_the_seed() {
        let plan = plan((150, 1200));
        let generator = generator(&plan);
        let target = &plan.targets[1];
        let first = generator.request(1, 77, target);
        let again = generator.request(1, 77, target);
        assert_eq!(
            (first.id.as_str(), &first.body),
            (again.id.as_str(), &again.body)
        );
        let other_key = Generator::new([7u8; 32], plan.words.clone()).request(1, 77, target);
        assert_ne!(first.id, other_key.id);
        assert_ne!(first.body, other_key.body);
    }
}
