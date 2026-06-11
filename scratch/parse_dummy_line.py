import re

def clean_ranks(word):
    # Map misreads to standard ranks
    misreads = {
        "0": "Q", "O": "Q", "D": "Q",
        "10": "T",
        "S": "T",
        "N": "K", "W": "K",
        "Z": "2",
        "E": "6",
        "I": "", # often Spade symbol or noise
        "F": "", # often Spade symbol or noise
        "B": "", # often Club symbol or noise
        "M": "", # often Heart/Diamond symbol or noise
        "P": "",
        "Q": "Q", "A": "A", "K": "K", "J": "J", "T": "T"
    }
    
    # Process the word
    cleaned = []
    # Replace known combinations first
    word = word.upper().replace("10", "T")
    
    for char in word:
        if char.isdigit():
            cleaned.append(char)
        elif char in misreads:
            repl = misreads[char]
            if repl:
                cleaned.append(repl)
        elif char in ["A", "K", "Q", "J", "T"]:
            cleaned.append(char)
            
    # Filter out duplicates and keep order
    final = []
    for c in cleaned:
        if c not in final:
            final.append(c)
    return final

def parse_dummy_line(text):
    # Split text by spaces
    words = text.split()
    print(f"Original words: {words}")
    
    # We expect 4 words for the 4 suits, but sometimes clubs is split or there is noise
    # Let's filter words to only those containing some ranks
    valid_words = []
    for w in words:
        cleaned = clean_ranks(w)
        if cleaned:
            valid_words.append((w, cleaned))
            
    print("Parsed suits:")
    suits_order = ["spade", "heart", "diamond", "club"]
    for idx, (orig, cleaned) in enumerate(valid_words[:4]):
        suit_name = suits_order[idx] if idx < 4 else f"extra_{idx}"
        print(f"  {suit_name.capitalize()}: {', '.join(cleaned)} (from '{orig}')")

def main():
    samples = [
        "IQ65ANMZ3AIJIEB6AJS9 4",
        "FQ65ANMW3Z3AJEB6AJS9 4",
        "FQ65ANMW3Z3AJEB6AJS9 4",
        "IQ65AMW3Z3AIJIEB6HAJS9 4",
        "IQ65ANM3Z3AJTEB6AJS9 4"
    ]
    for s in samples:
        print(f"\n--- Testing sample: '{s}' ---")
        parse_dummy_line(s)

if __name__ == "__main__":
    main()
