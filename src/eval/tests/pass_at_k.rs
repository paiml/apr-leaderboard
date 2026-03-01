use super::super::*;

#[test]
fn test_pass_at_k_all_pass() {
    // All 10 completions pass → pass@1 = 1.0
    assert!((pass_at_k(10, 10, 1) - 1.0).abs() < 1e-10);
    assert!((pass_at_k(10, 10, 10) - 1.0).abs() < 1e-10);
}

#[test]
fn test_pass_at_k_none_pass() {
    assert!((pass_at_k(10, 0, 1) - 0.0).abs() < 1e-10);
    assert!((pass_at_k(10, 0, 10) - 0.0).abs() < 1e-10);
}

#[test]
fn test_pass_at_k_one_of_ten() {
    // n=10, c=1, k=1: pass@1 = 1 - C(9,1)/C(10,1) = 1 - 9/10 = 0.1
    assert!((pass_at_k(10, 1, 1) - 0.1).abs() < 1e-10);
}

#[test]
fn test_pass_at_k_half_pass() {
    // n=10, c=5, k=1: pass@1 = 1 - C(5,1)/C(10,1) = 1 - 5/10 = 0.5
    assert!((pass_at_k(10, 5, 1) - 0.5).abs() < 1e-10);
}

#[test]
fn test_pass_at_k_known_value_k10() {
    // n=200, c=100, k=10: pass@10 should be very close to 1.0
    let result = pass_at_k(200, 100, 10);
    assert!(result > 0.999, "pass@10 with 50% pass rate should be >0.999, got {result}");
}

#[test]
fn test_pass_at_k_single_completion() {
    // n=1, c=1, k=1 → 1.0
    assert!((pass_at_k(1, 1, 1) - 1.0).abs() < 1e-10);
    // n=1, c=0, k=1 → 0.0
    assert!((pass_at_k(1, 0, 1) - 0.0).abs() < 1e-10);
}

#[test]
fn test_pass_at_k_k_equals_n() {
    // k=n: if c >= 1, guaranteed to include at least one passing
    // n=5, c=1, k=5: pass@5 = 1 - C(4,5)/C(5,5) = 1 - 0 = 1.0
    assert!((pass_at_k(5, 1, 5) - 1.0).abs() < 1e-10);
}

#[test]
fn test_pass_at_k_monotonic_in_k() {
    // pass@k should be non-decreasing in k
    let p1 = pass_at_k(100, 30, 1);
    let p5 = pass_at_k(100, 30, 5);
    let p10 = pass_at_k(100, 30, 10);
    assert!(p1 <= p5, "pass@1 ({p1}) > pass@5 ({p5})");
    assert!(p5 <= p10, "pass@5 ({p5}) > pass@10 ({p10})");
}

#[test]
fn test_pass_at_k_monotonic_in_c() {
    // pass@k should be non-decreasing in c
    let p0 = pass_at_k(100, 10, 5);
    let p1 = pass_at_k(100, 50, 5);
    let p2 = pass_at_k(100, 90, 5);
    assert!(p0 <= p1, "more correct should yield higher pass@k");
    assert!(p1 <= p2, "more correct should yield higher pass@k");
}

#[test]
fn test_pass_at_k_edge_k_zero() {
    assert!((pass_at_k(10, 5, 0) - 0.0).abs() < 1e-10);
}

#[test]
fn test_pass_at_k_large_n() {
    // Test with large numbers to verify no overflow
    let result = pass_at_k(1000, 500, 100);
    assert!(result > 0.999, "large n with 50% pass rate and k=100 should be ~1.0");
}

#[test]
fn test_average_pass_at_k() {
    // Two problems: one all-pass, one all-fail → average = 0.5
    let results = vec![(10, 10), (10, 0)];
    assert!((average_pass_at_k(&results, 1) - 0.5).abs() < 1e-10);
}

#[test]
fn test_average_pass_at_k_empty() {
    assert!((average_pass_at_k(&[], 1) - 0.0).abs() < 1e-10);
}

#[test]
fn test_average_pass_at_k_uniform() {
    // Three problems each with 5/10 pass → average pass@1 = 0.5
    let results = vec![(10, 5), (10, 5), (10, 5)];
    assert!((average_pass_at_k(&results, 1) - 0.5).abs() < 1e-10);
}
