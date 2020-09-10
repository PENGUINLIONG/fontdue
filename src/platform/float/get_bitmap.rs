#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

//#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn get_bitmap_default(a: &[f32], out: &mut [u8]) {
    debug_assert!(a.len() == out.len());
    let mut height = 0.0;
    for i in 0..a.len() {
        unsafe {
            height += a.get_unchecked(i);
            *(out.get_unchecked_mut(i)) = ((height) * 255.9) as u8;
        }
    }
}
#[target_feature(enable="sse,sse2")]
unsafe fn get_bitmap_sse2(a: &[__m128], out: &mut [i32]) {
    debug_assert!(a.len() == out.len());

    // offset = Zeroed out lanes
    let mut offset = _mm_setzero_ps();
    let zero = _mm_setzero_si128();
    for i in 0..a.len() {
        // Note that the casting to `__m128`s automatically reversed the byte
        // order.
        // x = Read 4 floats from self.a
        let mut x = *a.get_unchecked(i);
        // x += (0.0, x[0], x[1], x[2])
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
        // x += (0.0, 0.0, x[0], x[1])
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
        // x += offset
        x = _mm_add_ps(x, offset);

        // y = x * 255.9
        let y = _mm_mul_ps(x, _mm_set1_ps(255.9));
        // y = Convert y to i32s and truncate
        let mut y = _mm_cvttps_epi32(y);
        // y = Take the first byte of each of the 4 values in y and pack them into
        // the first 4 bytes of y.
        y = _mm_packus_epi16(_mm_packs_epi32(y, zero), zero);

        // Store the first 4 u8s from y in output.
        let pointer: &mut i32 = core::mem::transmute(out.get_unchecked_mut(i));
        *pointer = _mm_cvtsi128_si32(y);
        // offset = (x[3], x[3], x[3], x[3])
        offset = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(x), !0));
    }
}
#[target_feature(enable="avx,avx2")]
unsafe fn get_bitmap_avx2(a: &[__m256], out: &mut [i64]) {
    debug_assert!(a.len() == out.len());

    let mut offset = _mm256_setzero_ps();
    let zero = _mm256_setzero_si256();
    for i in 0..a.len() {
        let pointer = a.get_unchecked(i) as *const __m256 as *const f32;
        let mut x = _mm256_load_ps(pointer);
        x = _mm256_add_ps(x, _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 4)));
        x = _mm256_add_ps(x, _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 8)));
        x = _mm256_add_ps(x, _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 16)));
        x = _mm256_add_ps(x, offset);

        let y = _mm256_mul_ps(x, _mm256_set1_ps(255.9));
        let mut y = _mm256_cvttps_epi32(y);
        y = _mm256_packus_epi16(_mm256_packs_epi32(y, zero), zero);

        let pointer: &mut i32 = core::mem::transmute(out.get_unchecked_mut(i));
        *pointer = _mm256_cvtsi256_si32(y);
        offset = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(x), !0));
    }
}

pub fn get_bitmap(a: &[f32], out: &mut [u8]) {
    unsafe {
        //if cfg!(all(target_feature="sse", target_feature="sse2")) {
        if false {
            let (pre_a, a128, post_a) = a.align_to::<__m128>();
            let (pre_out, out128, post_out) = out.align_to_mut::<i32>();
            get_bitmap_default(pre_a, pre_out);
            //if cfg!(all(target_feature="avx", target_feature="avx2")) {
            if false {
                let (pre_a128, a256, post_a128) = a128.align_to::<__m256>();
                let (pre_out128, out256, post_out128) = out128.align_to_mut::<i64>();
                get_bitmap_sse2(pre_a128, pre_out128);
                get_bitmap_avx2(a256, out256);
                get_bitmap_sse2(post_a128, post_out128);
            } else {
                get_bitmap_sse2(a128, out128);
            }
            get_bitmap_default(post_a, post_out);
        } else {
            get_bitmap_default(a, out);
        }
    }
}
