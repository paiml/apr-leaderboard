#!/usr/bin/env bash
# Generate distillation prompts targeting 7B model weaknesses.
# Analyzes HumanEval failures and generates prompts in weak categories.
#
# Usage: scripts/generate-distill-prompts.sh [OUTPUT_FILE]
# Output: JSONL with {"prompt": "...", "source": "...", "kind": "..."}

set -euo pipefail

OUTPUT="${1:-data/distill/distill-prompts.jsonl}"
mkdir -p "$(dirname "$OUTPUT")"

echo "Generating distillation prompts → $OUTPUT"

# Categories derived from 7B HumanEval failures:
# - String manipulation: palindrome, encoding, nesting, fix_spaces, file_name_check
# - Mathematical reasoning: prime_fib, is_multiply_prime, poly/find_zero, tri sequence
# - List/array operations: remove_duplicates, max_fill, maximum, minPath, order_by_points
# - Number theory: iscube, is_simple_power, starts_one_ends, intersection
# - Complex logic: do_algebra, cycpattern_check, can_arrange

cat > "$OUTPUT" << 'PROMPTS'
{"prompt": "Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `make_palindrome(s: str) -> str` that finds the shortest palindrome that begins with the given string. Find the longest postfix that is a palindrome, then append the reverse of the prefix before that palindrome to the end.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `encode_shift(s: str) -> str` that encodes a string by shifting every character by 5 positions in the alphabet, wrapping around. Then write `decode_shift(s: str) -> str` to reverse it.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `is_nested(s: str) -> bool` that takes a string of only '[' and ']' brackets and returns True if there is a valid nested subsequence (at least one '[' followed by a ']' that contains another '[...]' inside).", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `fix_spaces(text: str) -> str` that replaces spaces: single space becomes '_', 2+ consecutive spaces become '-'.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `file_name_check(file_name: str) -> str` that returns 'Yes' if the file name is valid (starts with letter, has exactly one dot, extension is txt/exe/dll, digits count <= 3) or 'No' otherwise.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `cycpattern_check(a: str, b: str) -> bool` that returns True if any rotation of string b is a substring of string a.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `encode(message: str) -> str` that swaps the case of all letters and replaces all vowels with the vowel 2 positions ahead in the alphabet (a->c, e->g, i->k, o->q, u->w).", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `reverse_words(s: str) -> str` that reverses the order of words in a string while preserving whitespace.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `compress_string(s: str) -> str` that implements run-length encoding: 'aaabbc' -> 'a3b2c1'.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `longest_common_prefix(strs: list) -> str` that finds the longest common prefix among a list of strings.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `balanced_brackets(s: str) -> bool` that checks if a string of brackets ()[]{}  is properly balanced.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `prime_fib(n: int) -> int` that returns the n-th number that is both a Fibonacci number and prime. Example: prime_fib(1) = 2, prime_fib(2) = 3, prime_fib(3) = 5, prime_fib(4) = 13, prime_fib(5) = 89.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `is_multiply_prime(a: int) -> bool` that returns True if the given number is the multiplication of exactly 3 prime numbers (not necessarily distinct). Example: is_multiply_prime(30) returns True (2*3*5).", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `find_zero(xs: list) -> float` that finds a zero of a polynomial with coefficients xs using bisection method. The polynomial has an even number of coefficients and the largest non-zero coefficient guarantees a zero exists.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `tri(n: int) -> list` that returns the first n+1 numbers of the Tribonacci-like sequence where tri(1)=3, and for even n: tri(n) = 1 + n/2, for odd n: tri(n) = tri(n-1) + tri(n-2) + tri(n+1).", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `is_simple_power(x: int, n: int) -> bool` that returns True if x is a simple power of n (x = n^k for some non-negative integer k). Handle edge cases: x=1 is always True, n=1 means x must be 1.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `iscube(a: int) -> bool` that takes an integer and returns True if it is a perfect cube (including negative numbers). Example: iscube(1) = True, iscube(-1) = True, iscube(64) = True, iscube(2) = False.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `starts_one_ends(n: int) -> int` that returns the count of n-digit positive integers that start or end with 1.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `gcd(a: int, b: int) -> int` using Euclid's algorithm. Then write `lcm(a: int, b: int) -> int` using the GCD.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `sieve_of_eratosthenes(n: int) -> list` that returns all prime numbers up to n.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `prime_factors(n: int) -> list` that returns the prime factorization of n as a sorted list (with repetitions). Example: prime_factors(12) = [2, 2, 3].", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `modular_exponentiation(base: int, exp: int, mod: int) -> int` using fast exponentiation (repeated squaring).", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `catalan_number(n: int) -> int` that computes the n-th Catalan number using dynamic programming.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `remove_duplicates(numbers: list) -> list` that removes elements that appear more than once, keeping only elements that appear exactly once, preserving order.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `max_fill(grid: list, capacity: int) -> int` where grid is a 2D list of 0s and 1s representing water in wells. Each row is a well. Return the minimum number of bucket trips to empty all wells, where each trip removes 'capacity' units.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `maximum(arr: list, k: int) -> list` that returns the k largest elements from array arr, sorted in ascending order.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `solution(lst: list) -> int` that returns the sum of all odd-valued elements at even indices (0-indexed).", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `minPath(grid: list, k: int) -> list` that finds the minimum path of length k in an NxN grid. You can move to adjacent cells (up, down, left, right). Return the ordered list of values along the path, choosing lexicographically smallest.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `order_by_points(nums: list) -> list` that sorts a list of integers based on the sum of their digits. For equal digit sums, preserve original order.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `can_arrange(arr: list) -> int` that returns the largest index i such that arr[i] < arr[i-1]. If no such index exists, return -1.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `flatten(nested_list: list) -> list` that flattens an arbitrarily nested list. Example: flatten([1, [2, [3, 4], 5]]) = [1, 2, 3, 4, 5].", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `merge_sorted_lists(list1: list, list2: list) -> list` that merges two sorted lists into one sorted list without using sort().", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `rotate_list(lst: list, k: int) -> list` that rotates a list k positions to the right.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `sliding_window_max(nums: list, k: int) -> list` that returns the maximum value in each sliding window of size k.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `intersection(interval1: tuple, interval2: tuple) -> str` that finds the intersection of two closed intervals [a,b] and [c,d]. If the intersection length is prime, return 'YES', otherwise 'NO'. If no intersection, return 'NO'.", "source": "distill-number", "kind": "number_theory"}
{"prompt": "Write a Python function `check_dict_case(d: dict) -> bool` that returns True if all dictionary keys are either all lowercase strings or all uppercase strings. Empty dict returns False.", "source": "distill-edge", "kind": "edge_cases"}
{"prompt": "Write a Python function `even_odd_count(num: int) -> tuple` that returns a tuple (even_count, odd_count) counting even and odd digits in the absolute value of num.", "source": "distill-edge", "kind": "edge_cases"}
{"prompt": "Write a Python function `generate_integers(a: int, b: int) -> list` that returns even digits between a and b (inclusive, in ascending order). Only considers single even digits (2, 4, 6, 8). a and b can be in any order.", "source": "distill-edge", "kind": "edge_cases"}
{"prompt": "Write a Python function `do_algebra(operator: list, operand: list) -> int` that evaluates an algebraic expression. operator is a list of strings like ['+', '*', '-'] and operand is a list of non-negative integers. Example: do_algebra(['+', '*', '-'], [2, 3, 4, 5]) = 9 because (2+3)*4-5 = 15.", "source": "distill-logic", "kind": "complex_logic"}
{"prompt": "Write a Python function `longest_increasing_subsequence(nums: list) -> int` that returns the length of the longest strictly increasing subsequence using dynamic programming.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `knapsack(weights: list, values: list, capacity: int) -> int` that solves the 0/1 knapsack problem using dynamic programming.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `edit_distance(s1: str, s2: str) -> int` that computes the minimum edit distance (Levenshtein distance) between two strings.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `coin_change(coins: list, amount: int) -> int` that returns the fewest number of coins needed to make up the amount, or -1 if impossible.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `matrix_chain_order(dims: list) -> int` that finds the minimum number of scalar multiplications needed to multiply a chain of matrices.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `binary_search(arr: list, target: int) -> int` that returns the index of target in a sorted array, or -1 if not found.", "source": "distill-algo", "kind": "algorithms"}
{"prompt": "Write a Python function `merge_sort(arr: list) -> list` that implements merge sort.", "source": "distill-algo", "kind": "algorithms"}
{"prompt": "Write a Python function `quickselect(arr: list, k: int) -> int` that finds the k-th smallest element using the quickselect algorithm (average O(n)).", "source": "distill-algo", "kind": "algorithms"}
{"prompt": "Write a Python function `topological_sort(graph: dict) -> list` that returns a topological ordering of a DAG represented as an adjacency list.", "source": "distill-algo", "kind": "algorithms"}
{"prompt": "Write a Python function `dijkstra(graph: dict, start: str) -> dict` that computes shortest paths from start to all other nodes in a weighted graph.", "source": "distill-algo", "kind": "algorithms"}
{"prompt": "Write a Python function `trie_insert_search(words: list, query: str) -> bool` that builds a trie from words and searches for query. Return True if found.", "source": "distill-ds", "kind": "data_structures"}
{"prompt": "Write a Python class `MinHeap` with methods: insert(val), extract_min(), peek(), and __len__. Implement using a list without heapq.", "source": "distill-ds", "kind": "data_structures"}
{"prompt": "Write a Python class `LRUCache` with capacity, get(key), and put(key, value) methods. get and put should be O(1). Use OrderedDict.", "source": "distill-ds", "kind": "data_structures"}
{"prompt": "Write a Python function `inorder_traversal(root) -> list` for a binary tree node class with left, right, val attributes. Return values in inorder.", "source": "distill-ds", "kind": "data_structures"}
{"prompt": "Write a Python function `is_valid_bst(root) -> bool` that checks if a binary tree is a valid binary search tree.", "source": "distill-ds", "kind": "data_structures"}
{"prompt": "Write a Python function `two_sum(nums: list, target: int) -> list` that returns indices of two numbers that add up to target. Use a hash map for O(n) time.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `max_subarray(nums: list) -> int` that finds the contiguous subarray with the largest sum (Kadane's algorithm).", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `valid_parentheses(s: str) -> bool` that checks if a string of ()[]{}  has valid matching brackets.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `group_anagrams(strs: list) -> list` that groups strings that are anagrams of each other.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `product_except_self(nums: list) -> list` that returns an array where each element is the product of all other elements, without using division.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `three_sum(nums: list) -> list` that finds all unique triplets that sum to zero.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `trap_rain_water(height: list) -> int` that computes how much water can be trapped between bars of given heights.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `longest_substring_without_repeat(s: str) -> int` that finds the length of the longest substring without repeating characters.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `median_of_two_sorted(nums1: list, nums2: list) -> float` that finds the median of two sorted arrays in O(log(min(m,n))) time.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `serialize_tree(root) -> str` and `deserialize_tree(data: str)` for a binary tree using preorder traversal with '#' for null nodes.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `max_profit(prices: list) -> int` that finds the maximum profit from buying and selling a stock once (buy low, sell high, buy must be before sell).", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `climb_stairs(n: int) -> int` that returns the number of ways to climb n stairs taking 1 or 2 steps at a time.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `word_break(s: str, word_dict: list) -> bool` that determines if s can be segmented into words from word_dict.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `num_islands(grid: list) -> int` that counts the number of islands in a 2D grid of '1's (land) and '0's (water). Connected horizontally/vertically.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `min_window(s: str, t: str) -> str` that finds the minimum window substring of s that contains all characters of t.", "source": "distill-interview", "kind": "interview"}
{"prompt": "Write a Python function `permutations(nums: list) -> list` that returns all permutations of a list of distinct integers.", "source": "distill-combinatorics", "kind": "combinatorics"}
{"prompt": "Write a Python function `combinations(nums: list, k: int) -> list` that returns all combinations of k elements from nums.", "source": "distill-combinatorics", "kind": "combinatorics"}
{"prompt": "Write a Python function `subsets(nums: list) -> list` that returns all possible subsets (the power set) of a list of distinct integers.", "source": "distill-combinatorics", "kind": "combinatorics"}
{"prompt": "Write a Python function `generate_parentheses(n: int) -> list` that generates all valid combinations of n pairs of parentheses.", "source": "distill-combinatorics", "kind": "combinatorics"}
{"prompt": "Write a Python function `solve_n_queens(n: int) -> list` that returns all solutions to the N-Queens puzzle as lists of queen positions.", "source": "distill-combinatorics", "kind": "combinatorics"}
{"prompt": "Write a Python function `count_inversions(arr: list) -> int` that counts the number of inversions using merge sort (pair where i < j but arr[i] > arr[j]).", "source": "distill-algo", "kind": "algorithms"}
{"prompt": "Write a Python function `kmp_search(text: str, pattern: str) -> list` that returns all starting indices where pattern occurs in text, using the KMP algorithm.", "source": "distill-algo", "kind": "algorithms"}
{"prompt": "Write a Python function `matrix_multiply(a: list, b: list) -> list` that multiplies two 2D matrices.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `determinant(matrix: list) -> float` that computes the determinant of a square matrix using cofactor expansion.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `newton_sqrt(n: float, epsilon: float = 1e-10) -> float` that computes the square root of n using Newton's method.", "source": "distill-math", "kind": "mathematical_reasoning"}
{"prompt": "Write a Python function `roman_to_int(s: str) -> int` that converts a Roman numeral string to an integer.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `int_to_roman(num: int) -> str` that converts an integer (1 to 3999) to a Roman numeral string.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `atoi(s: str) -> int` that converts a string to an integer, handling whitespace, optional sign, overflow (clamp to [-2^31, 2^31-1]), and stopping at non-digit characters.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `zigzag_conversion(s: str, num_rows: int) -> str` that converts a string into zigzag pattern on num_rows rows and reads it off row by row.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `decode_string(s: str) -> str` that decodes a run-length encoded string like '3[a2[bc]]' -> 'abcbcabcbcabcbc'.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `longest_palindrome_substring(s: str) -> str` that finds the longest palindromic substring using expand-around-center approach.", "source": "distill-string", "kind": "string_manipulation"}
{"prompt": "Write a Python function `next_permutation(nums: list) -> None` that modifies nums in-place to the next lexicographic permutation.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `spiral_order(matrix: list) -> list` that returns elements of a 2D matrix in spiral order.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `rotate_matrix(matrix: list) -> None` that rotates an NxN matrix 90 degrees clockwise in-place.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `set_matrix_zeroes(matrix: list) -> None` that sets entire row and column to 0 if an element is 0, in-place.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `find_peak_element(nums: list) -> int` that finds a peak element (greater than neighbors) in O(log n) time.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `search_rotated(nums: list, target: int) -> int` that searches for target in a rotated sorted array in O(log n).", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `interval_merge(intervals: list) -> list` that merges overlapping intervals. Input: list of [start, end] pairs.", "source": "distill-list", "kind": "list_operations"}
{"prompt": "Write a Python function `longest_common_subsequence(s1: str, s2: str) -> int` that returns the length of the longest common subsequence.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `min_path_sum(grid: list) -> int` that finds the minimum path sum from top-left to bottom-right in a grid, moving only right or down.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `unique_paths(m: int, n: int) -> int` that counts the number of unique paths from top-left to bottom-right in an m x n grid.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `max_product_subarray(nums: list) -> int` that finds the contiguous subarray with the largest product.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `decode_ways(s: str) -> int` that counts the number of ways to decode a digit string where 'A'=1...'Z'=26.", "source": "distill-dp", "kind": "dynamic_programming"}
{"prompt": "Write a Python function `rob_houses(nums: list) -> int` that finds the maximum amount you can rob from non-adjacent houses.", "source": "distill-dp", "kind": "dynamic_programming"}
PROMPTS

COUNT=$(wc -l < "$OUTPUT")
echo "Generated $COUNT distillation prompts → $OUTPUT"
echo "Categories: string_manipulation, mathematical_reasoning, list_operations, number_theory, edge_cases, complex_logic, dynamic_programming, algorithms, data_structures, interview, combinatorics"
