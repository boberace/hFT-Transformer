#! /bin/bash
set -e

## GuitarSet (Zenodo record 3371780): annotation + audio_mono-mic
## run from the repository root: bash corpus/guitar/EXE-CORPUS-GUITARSET.sh
CURRENT_DIR=$(pwd)
D=$CURRENT_DIR/corpus/GuitarSet
CONFIG=$CURRENT_DIR/corpus/config_guitar.json
PYTHON=${PYTHON:-python3}

# 1. download GuitarSet and expand it
mkdir -p $D/zip
for NAME in annotation audio_mono-mic; do
    FILE=$D/zip/$NAME.zip
    if test -f "$FILE"; then
        echo "$FILE exists, proceed to unzip"
    else
        echo "$FILE does not exist. Downloading..."
        curl -L --fail -o $FILE.part https://zenodo.org/api/records/3371780/files/$NAME.zip/content
        mv $FILE.part $FILE
    fi
    if test ! -d "$D/$NAME"; then
        unzip -q $FILE -d $D/$NAME.part
        mv $D/$NAME.part $D/$NAME
    fi
done
# (the zips may or may not contain a top-level folder)
D_ANNOTATION=$(dirname $(find $D/annotation -name '*.jams' | head -1))
D_AUDIO=$(dirname $(find $D/audio_mono-mic -name '*_mic.wav' | head -1))

# 2. make lists (leave-one-player-out: player 05 test, player 04 valid, rest train) and link files
mkdir -p $D/list $D/wav $D/jams
$PYTHON $CURRENT_DIR/corpus/guitar/make_list_guitarset.py -d_annotation $D_ANNOTATION -d_audio $D_AUDIO -d_wav $D/wav -d_jams $D/jams -d_list $D/list -player_valid 04 -player_test 05

# 3. convert wav to log-mel spectrogram
mkdir -p $D/feature
$PYTHON $CURRENT_DIR/corpus/guitar/run_compat.py $CURRENT_DIR/corpus/conv_wav2fe.py -d_list $D/list -d_wav $D/wav -d_feature $D/feature -config $CONFIG

# 4. convert jams to note
mkdir -p $D/note
$PYTHON $CURRENT_DIR/corpus/guitar/conv_jams2note.py -d_list $D/list -d_jams $D/jams -d_note $D/note -config $CONFIG

# 5. convert note to label (pitch + string/fret)
mkdir -p $D/label
$PYTHON $CURRENT_DIR/corpus/guitar/conv_note2label_guitar.py -d_list $D/list -d_note $D/note -d_label $D/label -config $CONFIG

# 6. convert note to reference for evaluation
mkdir -p $D/reference
$PYTHON $CURRENT_DIR/corpus/guitar/run_compat.py $CURRENT_DIR/corpus/conv_note2ref.py -f_list $D/list/valid.list -d_note $D/note -d_ref $D/reference
$PYTHON $CURRENT_DIR/corpus/guitar/run_compat.py $CURRENT_DIR/corpus/conv_note2ref.py -f_list $D/list/test.list -d_note $D/note -d_ref $D/reference

# 7. make dataset
mkdir -p $D/dataset
$PYTHON $CURRENT_DIR/corpus/guitar/make_dataset_guitar.py -f_config_in $CONFIG -f_config_out $D/dataset/config.json -d_dataset $D/dataset -d_list $D/list -d_feature $D/feature -d_label $D/label
