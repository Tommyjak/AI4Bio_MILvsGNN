# TODO list for MIL_utils
## pairings.py
* The path to follow to access data is hardcoded: separate the path to the data from the code, and make it an argument to the function.
* No filtering of filler parts of interactions is done (we should be able to filter them out)

## general stuff
* EXCLUDED_SESSIONS = {342, 394, 398, 460, 451, 458, 480}
* Paths changed from ../daic-woz to ./daic-woz — this means they are now relative to wherever you run the script from, not relative to MIL_utils/. This is correct only if you run from AI4Bio/. If you run from MIL_utils/ it will break. On a remote server you'll likely cd into MIL_utils/ and run python3 train.py — so paths need to be either absolute or relative to MIL_utils/. Fix: use absolute paths on the server, or make ds_yaml_parser.py resolve paths relative to the yaml file location
* Complete a successful smoke test before running the training on aimagelab server

## model.py
Architecturally correct. One concern for the server: InstanceEncoder.__init__ calls AutoModel.from_pretrained which downloads MentalBERT (~440MB) at runtime. On a server without internet access this will crash. Pre-download the model before the job starts:
bash# run this locally or in an interactive session before queuing

python3 -c "from transformers import AutoModel, AutoTokenizer; \
            AutoModel.from_pretrained('mental/mental-bert-base-uncased'); \
            AutoTokenizer.from_pretrained('mental/mental-bert-base-uncased')"

This caches it in ~/.cache/huggingface/ which persists across jobs.
