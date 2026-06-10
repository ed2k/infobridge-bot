import json
import os

def main():
    log_path = "/Users/admin/.gemini/antigravity/brain/fbe112da-b653-45be-bc88-dea616da3ad1/.system_generated/logs/transcript.jsonl"
    if not os.path.exists(log_path):
        print(f"Log not found at {log_path}")
        return
        
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("type") == "USER_INPUT":
                    print(f"User: {data.get('content')}")
                elif data.get("source") == "MODEL" and data.get("type") == "PLANNER_RESPONSE":
                    content = data.get("content", "")
                    if "dummy" in content.lower():
                        # print model response lines containing dummy
                        print(f"Model: ...")
                        for l in content.split('\n'):
                            if any(w in l.lower() for w in ["dummy", "north", "spade", "heart", "club", "diamond"]):
                                print(f"  {l}")
            except Exception as e:
                pass

if __name__ == "__main__":
    main()
