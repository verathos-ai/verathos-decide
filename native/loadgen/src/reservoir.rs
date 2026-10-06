//! Uniform sampling of a fixed number of items from a stream of unknown length (Algorithm R).

use rand::RngExt;
use rand_chacha::ChaCha12Rng;

pub struct Reservoir<T> {
    capacity: usize,
    seen: u64,
    items: Vec<T>,
    rng: ChaCha12Rng,
}

impl<T> Reservoir<T> {
    pub fn new(capacity: usize, rng: ChaCha12Rng) -> Self {
        Self {
            capacity,
            seen: 0,
            items: Vec::with_capacity(capacity),
            rng,
        }
    }

    /// Offers the next item of the stream. `make` runs only if the item is kept.
    pub fn offer(&mut self, make: impl FnOnce() -> T) {
        self.seen += 1;
        if self.items.len() < self.capacity {
            self.items.push(make());
            return;
        }
        let slot = self.rng.random_range(0..self.seen);
        if let Some(item) = usize::try_from(slot)
            .ok()
            .and_then(|slot| self.items.get_mut(slot))
        {
            *item = make();
        }
    }

    pub fn take(&mut self) -> Vec<T> {
        std::mem::take(&mut self.items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn reservoir(capacity: usize, seed: u64) -> Reservoir<u32> {
        Reservoir::new(capacity, ChaCha12Rng::seed_from_u64(seed))
    }

    #[test]
    fn keeps_everything_until_full_then_exactly_capacity() {
        let mut short = reservoir(10, 1);
        (0..4).for_each(|item| short.offer(|| item));
        assert_eq!(short.take(), vec![0, 1, 2, 3]);

        let mut long = reservoir(10, 1);
        (0..1_000).for_each(|item| long.offer(|| item));
        let kept = long.take();
        assert_eq!(kept.len(), 10);
        let mut distinct = kept.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 10);

        let mut none = reservoir(0, 1);
        (0..100).for_each(|item| none.offer(|| item));
        assert!(none.take().is_empty());
    }

    #[test]
    fn builds_only_the_items_it_keeps() {
        let mut reservoir = reservoir(3, 2);
        let mut built = 0;
        for item in 0..10_000 {
            reservoir.offer(|| {
                built += 1;
                item
            });
        }
        // Expected number of builds: 3 + sum over n in 4..=10000 of 3/n, about 25.
        assert!((3..=60).contains(&built), "{built} items built");
    }

    #[test]
    fn every_item_is_equally_likely() {
        const ITEMS: usize = 50;
        const CAPACITY: usize = 5;
        const TRIALS: u64 = 40_000;
        let mut counts = [0u32; ITEMS];
        for trial in 0..TRIALS {
            let mut reservoir = reservoir(CAPACITY, trial);
            (0..ITEMS as u32).for_each(|item| reservoir.offer(|| item));
            reservoir
                .take()
                .into_iter()
                .for_each(|item| counts[item as usize] += 1);
        }
        let expected = TRIALS as f64 * CAPACITY as f64 / ITEMS as f64;
        let chi_square: f64 = counts
            .iter()
            .map(|&count| (f64::from(count) - expected).powi(2) / expected)
            .sum();
        // 49 degrees of freedom: the 99.9th percentile of chi-square is about 85.4.
        assert!(
            chi_square < 85.4,
            "chi-square {chi_square:.1}, counts {counts:?}"
        );
    }
}
