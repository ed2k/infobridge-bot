def parse_ranks_into_suits(ranks_list):
    rank_order = {r: idx for idx, r in enumerate(["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"])}
    
    suits = []
    current_suit = []
    
    for r in ranks_list:
        if not current_suit:
            current_suit.append(r)
        else:
            prev_r = current_suit[-1]
            # If the new rank is higher than or equal to the previous rank, it must start a new suit
            if rank_order[r] <= rank_order[prev_r]:
                suits.append(current_suit)
                current_suit = [r]
            else:
                current_suit.append(r)
                
    if current_suit:
        suits.append(current_suit)
        
    suits_order = ["spade", "heart", "diamond", "club"]
    for idx, suit_cards in enumerate(suits):
        name = suits_order[idx] if idx < 4 else f"extra_{idx}"
        print(f"  {name.capitalize()}: {', '.join(suit_cards)}")

def main():
    # The cleaned rank sequence from the OCR string:
    # Q, 6, 5, A, K, 3, A, J, 6, A, J, 9, 4
    ranks = ["Q", "6", "5", "A", "K", "3", "A", "J", "6", "A", "J", "9", "4"]
    print("Parsing ranks:", ranks)
    parse_ranks_into_suits(ranks)

if __name__ == "__main__":
    main()
