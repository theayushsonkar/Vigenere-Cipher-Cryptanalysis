import re
import math
from collections import defaultdict, Counter
import string
import numpy as np
import json

def clean_input(text):
    # Keep only A-Z
    text = text.upper()
    text = re.sub(r'[^A-Z]', '', text)
    return text

def find_repeated_trigraphs(ciphertext):
    trigraph_positions = defaultdict(list)

    # Collect all trigraphs
    for i in range(len(ciphertext) - 2):
        trigram = ciphertext[i:i+3]
        trigraph_positions[trigram].append(i)

    # Keep only repeated trigraphs
    repeated = {t: pos for t, pos in trigraph_positions.items() if len(pos) > 1}
    return repeated

def get_lower_factors(num):
    factors = []
    for i in range(2, 6):
        if num % i == 0:
            factors.append(i)
    return factors

def compute_gcds(repeated_trigraphs):
    gcd_results = {}
    all_gcds = []

    for trigram, positions in repeated_trigraphs.items():
        distances = [positions[j] - positions[i]
                     for i in range(len(positions))
                     for j in range(i+1, len(positions))]

        if len(distances) == 1 and distances[0] > 5:
            distances = get_lower_factors(distances[0])

        if len(distances) > 1:
            # Compute GCD of all distances
            distances.sort()
            trigram_gcd = distances[0]
            for d in distances[1:]:
                trigram_gcd = math.gcd(trigram_gcd, d)

            gcd_results[trigram] = (distances, trigram_gcd)

            # Add to frequency list only if gcd <= 5 and > 0
            if 1 < trigram_gcd <= 5:
                all_gcds.append(trigram_gcd)
            else:
                lower_factors = get_lower_factors(trigram_gcd)
                for lf in lower_factors:
                    all_gcds.append(lf)

        elif len(distances) == 1 and distances[0] < 6:
            all_gcds.append(distances[0])
    # Count frequencies
    gcd_frequencies = Counter(all_gcds)
    return gcd_results, gcd_frequencies

#=================== Calculation of Index of Coincidence ===================
def Index_of_coincidence(text, top_2_gcds):
    
    best_key_length = 0
    min_ic_difference = float('inf')
    top_2_ics = {}
    
    for length in top_2_gcds:
        if length == 0:
            continue
        
        groups = {i: [] for i in range(length)}  # Initialize groups 0 to length-1
        # make groups
        for i, char in enumerate(text):
            group_num = i % length 
            groups[group_num].append(char)
        
        # calculate the average ICs
        total_ic = 0
        for group_num in range(length):
            sub_message = "".join(groups[group_num]) #Takes all characters in group & joins them in a string
            total_ic += calculate_ic(sub_message)
        
        average_ic = total_ic / length
        ic_difference = abs(average_ic - 0.0667)

        print("IC for length ",length,average_ic)
        top_2_ics[length] = average_ic
        # Find the key length with the closest average IC to English
        if ic_difference < min_ic_difference:
            min_ic_difference = ic_difference
            best_key_length = length

    return top_2_ics
        

def calculate_ic(text):
    freq = Counter(text)
    total_chars = len(text)
    
    ic_numerator = 0
    for count in freq.values():
        ic_numerator += count * (count - 1)
    
    ic_denominator = total_chars * (total_chars - 1)
    
    return ic_numerator / ic_denominator if ic_denominator > 0 else 0


def rank_key_lengths(gcd_counts, ic_values, candidates=None, w_gcd=0.5, w_ic=0.5):
    # gcd_counts: dict length -> count (int)
    # ic_values: dict length -> avg IC (float)
    if candidates is None:
        candidates = sorted(set(gcd_counts) | set(ic_values))
    # prepare arrays
    counts = [gcd_counts.get(L, 0) for L in candidates]
    # ics = [ic_values.get(L, 0.0) for L in candidates]

    # normalize gcd counts -> [0,1]
    min_c, max_c = min(counts), max(counts)
    denom_c = max(1e-9, (max_c - min_c))
    norm_gcd = {L: (gcd_counts.get(L,0) - min_c) / denom_c for L in candidates}

    # normalize IC closeness -> higher is better when closer to 0.065
    english_ic = 0.065
    devs = [abs(ic_values.get(L, english_ic) - english_ic) for L in candidates]
    max_dev = max(devs) if devs else 1e-9
    max_dev = max(max_dev, 1e-3)  # avoid tiny denom
    norm_ic = {}
    for L in candidates:
        d = abs(ic_values.get(L, english_ic) - english_ic)
        s = 1.0 - (d / max_dev)
        # clamp
        s = max(0.0, min(1.0, s))
        # penalty for suspiciously large IC (likely letter-bias)
        if ic_values.get(L, 0.0) > 0.12:
            s *= 0.5  # reduce trust if IC unusually large
        norm_ic[L] = s

    # adapt weights heuristics: if gcd top is huge compared to others, nudge weight
    sorted_counts = sorted(counts, reverse=True)
    if sorted_counts and sorted_counts[0] > 0:
        if len(sorted_counts) > 1 and sorted_counts[0] >= 3 * (sorted_counts[1] + 1):
            w_gcd = min(0.9, w_gcd + 0.2)  # strong gcd signal -> favor it

    # combined score
    scores = {}
    for L in candidates:
        scores[L] = w_gcd * norm_gcd[L] + w_ic * norm_ic[L]

    # rank
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked, norm_gcd, norm_ic


# MIC calculation
# Split ciphertext into substrings by key length
def split_into_substrings(ciphertext, key_length):
    substrings = ['' for _ in range(key_length)]
    for i, ch in enumerate(ciphertext):
        substrings[i % key_length] += ch
    return substrings

# Frequency distribution of letters in a string
def letter_frequencies(s):
    counts = Counter(s)
    total = len(s)
    return [counts.get(ch, 0) / total for ch in string.ascii_uppercase]

# Compute MIC between two substrings with shift
def mic(s1, s2, shift):
    f1 = letter_frequencies(s1)
    f2 = letter_frequencies(s2)
    # shift frequencies of s2
    f2_shifted = f2[-shift:] + f2[:-shift] if shift > 0 else f2
    return sum(f1[i] * f2_shifted[i] for i in range(26))

# Find relative shifts
def find_relative_shifts(ciphertext, key_length):
    substrings = split_into_substrings(ciphertext, key_length)

    results = {}
    for j in range(1, key_length):  # compare c1 with others
        best_shift, best_mic = 0, 0
        for shift in range(26):
            val = mic(substrings[0], substrings[j], shift)
            # print(f"FOR C1 vs C{j+1} for shift {shift} : {val}")
            if val > best_mic:
                best_mic, best_shift = val, shift
        results[f"C1 vs C{j+1}"] = (best_shift, best_mic)
    return results

# Vigenere decryption
def vigenere_decrypt(ciphertext, key):
    plaintext = []
    key_len = len(key)
    for i, ch in enumerate(ciphertext):
        if ch in string.ascii_uppercase:
            shift = ord(key[i % key_len]) - ord('A')
            p = (ord(ch) - ord('A') - shift) % 26
            plaintext.append(chr(p + ord('A')))
        else:
            plaintext.append(ch)
    return ''.join(plaintext)

# Generate candidate keys from relative shifts
def generate_keys(relative_shifts, key_length):
    keys = []
    for base in range(26):  # try all A-Z for C1
        key = [None] * key_length
        key[0] = chr((base % 26) + ord('A'))
        for j in range(1, key_length):
            shift = relative_shifts[j-1]  # relative shift C1→C(j+1)
            # If shift = k, then key[j] = (base + shift) mod 26
            key[j] = chr(((base - shift) % 26) + ord('A'))
        keys.append(''.join(key))
    return keys

# Expected English letter frequencies
english_freqs = {
    'A':0.08167, 'B':0.01492, 'C':0.02782, 'D':0.04253, 'E':0.12702,
    'F':0.02228, 'G':0.02015, 'H':0.06094, 'I':0.06966, 'J':0.00153,
    'K':0.00772, 'L':0.04025, 'M':0.02406, 'N':0.06749, 'O':0.07507,
    'P':0.01929, 'Q':0.00095, 'R':0.05987, 'S':0.06327, 'T':0.09056,
    'U':0.02758, 'V':0.00978, 'W':0.02360, 'X':0.00150, 'Y':0.01974,
    'Z':0.00074
}

def chi_squared_score(text):
    N = len(text)
    counts = Counter(text)
    score = 0
    for ch in string.ascii_uppercase:
        observed = counts.get(ch, 0)
        expected = english_freqs[ch] * N
        score += (observed - expected)**2 / expected if expected > 0 else 0
    return score

# ================= Dictionary Scoring =================
COMMON_WORDS = [
    "THE","AND","OF","TO","IN","THAT","IT","IS","WAS","HE","FOR",
    "ON","ARE","AS","WITH","HIS","THEY","AT","BE","THIS","HAVE",
    "FROM","OR","ONE","HAD","BY","WORD","BUT","NOT","WHAT","ALL"
]

def english_score(text):
    """Count occurrences of common English words in text (case-insensitive)."""
    score = 0
    upper_text = text.upper()
    for w in COMMON_WORDS:
        score += upper_text.count(w)
    return score


def normalize(values, invert=False):
    min_v, max_v = min(values), max(values)
    if invert:
        # lower is better, so invert
        return [(max_v - v) / (max_v - min_v + 1e-9) for v in values]
    else:
        # higher is better
        return [(v - min_v) / (max_v - min_v + 1e-9) for v in values]

def log_minmax_norm(vals):
    vals = np.array(vals, dtype=float)
    y = np.log(vals + 1.0)           # compress outliers
    y_min, y_max = y.min(), y.max()
    denom = max(1e-9, y_max - y_min)
    normalized_y = (y - y_min) / denom
    norm_chi_score = (1.0 - normalized_y).tolist()
    return norm_chi_score


def reconstruct_plaintext(ciphertext, decrypted_letters):
    """Reinsert spaces/punctuation/case into the decrypted text."""
    result = []
    idx = 0  # index into decrypted_letters (A–Z only)
    for ch in ciphertext:
        if ch.isalpha():
            # pick next decrypted letter
            plain_char = decrypted_letters[idx]
            # match case
            if ch.islower():
                plain_char = plain_char.lower()
            result.append(plain_char)
            idx += 1
        else:
            # leave punctuation, spaces, etc
            result.append(ch)
    return "".join(result)


def write_output_json(filename, key, plaintext):
    output_data = {
        "key": key,
        "plaintext": plaintext
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4) 


def main():
    # Read ciphertext from file
    with open("input2.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    ciphertext = clean_input(raw_text)
    repeated_trigraphs = find_repeated_trigraphs(ciphertext)
    gcd_results, gcd_frequencies = compute_gcds(repeated_trigraphs)

    print("\n=== GCD Frequencies (<= 5) ===")
    for gcd_val, freq in gcd_frequencies.most_common():
        print(f"GCD {gcd_val}: {freq} times")
    
    # Get top 2 gcds
    top_2_gcds = gcd_frequencies.most_common(2)
    top_2_gcds = dict(top_2_gcds)
    print(top_2_gcds)

    top_2_ics = Index_of_coincidence(ciphertext,top_2_gcds)
    ranked, ng, ni = rank_key_lengths(top_2_gcds, top_2_ics)
    print("Top 2 keylengths ",ranked)
    key_length = ranked[0][0]
    print("Key length ",key_length)

    relative_shifts = find_relative_shifts(ciphertext, key_length)
    shifts = []
    for pair, (shift, mic_val) in relative_shifts.items():
        print(f"{pair}: shift = {shift}, MIC = {mic_val:.4f}")
        shifts.append(shift)
    # shifts = [2,16,3,2]
    candidates = []
    for key in generate_keys(shifts, key_length):
        plaintext = vigenere_decrypt(ciphertext, key)
        # ic = index_of_coincidence(plaintext)
        ic = calculate_ic(plaintext)
        chi_score = chi_squared_score(plaintext)
        eng_score = english_score(plaintext)   # use dictionary scoring
        candidates.append((abs(ic - 0.066), key, ic, chi_score, eng_score, plaintext))

    # Sort by closeness to English IC
    candidates.sort(key=lambda x: x[3])

    # After computing chi_score and dict_score for each candidate
    chi_scores = [c[3] for c in candidates]
    dict_scores = [c[4] for c in candidates]
    print("chi scores ", chi_scores)
    norm_chi = log_minmax_norm(chi_scores)
    norm_dict = normalize(dict_scores, invert=False)  # higher dict is better

    # Assign final combined score
    final_candidates = []
    for i, (diff,key,ic,chi, d, text) in enumerate(candidates):
        score = 0.6 * norm_chi[i] + 0.4 * norm_dict[i]
        final_candidates.append((score, chi, d, key, text))

    # Sort by final score
    final_candidates.sort(key=lambda x: x[0], reverse=True)

    # Print top 3
    decrypted_letters = final_candidates[0][4]
    final_key = final_candidates[0][3]
    if(len(final_key) == 2 and final_key[0] == final_key[1]):
        final_key = final_key[0]
    for rank, (score, chi, d, key, text) in enumerate(final_candidates[:2], 1):
        print(f"Rank {rank}: Key={key}, Score={score:.3f}, Chi={chi:.1f}, Words={d}")
        print("Plaintext sample:", text[:200], "\n")

    
    plaintext_preserved = reconstruct_plaintext(raw_text, decrypted_letters)

    # save to JSON
    write_output_json("output.json", final_key, plaintext_preserved)


if __name__ == "__main__":
    main()



def run_from_text(raw_text: str):
    ciphertext = clean_input(raw_text)
    repeated_trigraphs = find_repeated_trigraphs(ciphertext)
    gcd_results, gcd_frequencies = compute_gcds(repeated_trigraphs)

    top_2_gcds = dict(gcd_frequencies.most_common(2))
    top_2_ics = Index_of_coincidence(ciphertext, top_2_gcds)
    ranked, _, _ = rank_key_lengths(top_2_gcds, top_2_ics)
    key_length = ranked[0][0]

    relative_shifts = find_relative_shifts(ciphertext, key_length)
    shifts = [shift for _, (shift, _) in relative_shifts.items()]

    candidates = []
    for key in generate_keys(shifts, key_length):
        plaintext = vigenere_decrypt(ciphertext, key)
        ic = calculate_ic(plaintext)
        chi_score = chi_squared_score(plaintext)
        eng_score = english_score(plaintext)
        candidates.append((abs(ic - 0.066), key, ic, chi_score, eng_score, plaintext))

    candidates.sort(key=lambda x: x[3])
    chi_scores = [c[3] for c in candidates]
    dict_scores = [c[4] for c in candidates]
    norm_chi = log_minmax_norm(chi_scores)
    norm_dict = normalize(dict_scores, invert=False)

    final_candidates = []
    for i, (diff,key,ic,chi,d,text) in enumerate(candidates):
        score = 0.6 * norm_chi[i] + 0.4 * norm_dict[i]
        final_candidates.append((score, chi, d, key, text))

    final_candidates.sort(key=lambda x: x[0], reverse=True)

    decrypted_letters = final_candidates[0][4]
    final_key = final_candidates[0][3]
    plaintext_preserved = reconstruct_plaintext(raw_text, decrypted_letters)

    return {"key": final_key, "plaintext": plaintext_preserved}
