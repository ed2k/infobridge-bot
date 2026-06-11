def parse_text(text):
    # Map characters
    # If a character is a suit symbol (like Heart/Diamond symbol being read as M, or Club symbol as B or 6), we ignore it.
    # Let's define the character mappings:
    mapping = {
        "0": "Q", "O": "Q", "D": "Q",
        "1": "T", "S": "T",
        "N": "K", "W": "K",
        "Z": "", # ignore Z
        "E": "6",
        "M": "", # Heart/Diamond symbol
        "B": "", # Club symbol
        "I": "",
        "F": "",
        "H": "",
        "X": "",
    }
    
    # Process
    cleaned = []
    text = text.upper().replace("10", "T")
    for char in text:
        if char.isdigit():
            cleaned.append(char)
        elif char in mapping:
            repl = mapping[char]
            if repl:
                cleaned.append(repl)
        elif char in ["A", "K", "Q", "J", "T"]:
            cleaned.append(char)
            
    # Remove adjacent duplicates
    filtered = []
    for c in cleaned:
        if not filtered or filtered[-1] != c:
            filtered.append(c)
            
    # Split into suits
    rank_order = {r: idx for idx, r in enumerate(["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"])}
    suits = []
    current_suit = []
    for r in filtered:
        if r not in rank_order:
            continue
        if not current_suit:
            current_suit.append(r)
        else:
            prev_r = current_suit[-1]
            if rank_order[r] <= rank_order[prev_r]:
                suits.append(current_suit)
                current_suit = [r]
            else:
                current_suit.append(r)
    if current_suit:
        suits.append(current_suit)
        
    return suits

def main():
    samples = [
        "IQ65AMWZ3AIJIEB6AJS9 4",
        "IQ65AMW3Z3AIJIEB6HAJS9 4",
        "IQ65ANM3Z3AJTEB6AJS9 4",
        "FQ65ANMW3Z3AJEB6AJS9 4",
        "FQ65AUMZ3AIJTEBHAJS9 4",
        "IQ65AUM3Z3AIJTBHAJS9 4"
    ]
    for s in samples:
        suits = parse_text(s)
        total = sum(len(st) for st in suits)
        print(f"\nText: '{s}' -> Total: {total} cards")
        suits_order = ["spade", "heart", "diamond", "club"]
        for idx, st in enumerate(suits[:4]):
            name = suits_order[idx] if idx < 4 else f"extra_{idx}"
            print(f"  {name.capitalize()}: {', '.join(st)}")

if __name__ == "__main__":
    main()
