import pandas as pd
import os

def build_pairs(path, min_token_length=1):
    df = pd.read_csv(path, sep="\t")

    pairs = []
    current_ellie = []
    current_participant = []

    for _, row in df.iterrows():
        speaker = row["speaker"]
        text = str(row["value"]).strip()

        if speaker == "Ellie":
            # If we have a complete exchange, save it
            if current_ellie and current_participant:
                ellie_text = " ".join(current_ellie)
                participant_text = " ".join(current_participant)
                # Filter out pairs where participant is just fillers
                if len(participant_text.split()) >= min_token_length:
                    pairs.append({
                        "ellie": ellie_text,
                        "participant": participant_text,
                        "instance": ellie_text + " [SEP] " + participant_text
                    })
                current_participant = []
                current_ellie = []
            current_ellie.append(text)

        elif speaker == "Participant":
            current_participant.append(text)

    # Don't forget the last exchange
    if current_ellie and current_participant:
        pairs.append({
            "ellie": " ".join(current_ellie),
            "participant": " ".join(current_participant),
            "instance": " ".join(current_ellie) + " [SEP] " + " ".join(current_participant)
        })

    return pd.DataFrame(pairs)

def run_total_pairing(ds_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for filename in sorted(os.listdir(ds_path)):
        if filename.endswith("_TRANSCRIPT.csv"):
            file_path = os.path.join(ds_path, filename)
            pairs_df = build_pairs(file_path)
            output_file = os.path.join(output_path, filename.replace("_TRANSCRIPT.csv", "_PAIRS.csv"))
            pairs_df.to_csv(output_file, index=False)
            print(f"{filename.split('_')[0]}: {len(pairs_df)} pairs")

def main():
    run_total_pairing("../daic-woz/transcripts", "../daic-woz/pairs")

if __name__ == "__main__":
    main()
    