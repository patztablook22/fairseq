# Then download and preprocess the data
set -e

OPS="id push pop shift unshift remove reverse duplicate flip"
WORKERS=4

EXP_DIR=

DATA_DIR="custom_examples/translation/bitedit"
VALID_SETS="20.valid"
TEST_SETS="20.test"

JOINED_DICT_OPT=


HELP=1
while [[ $# -gt 0 ]]; do
key="$1"
case $key in
    --expdir)
        EXP_DIR="$2"
        shift
    ;;
    --datadir)
        DATA_DIR="$2"
        shift
    ;;
    --ops)
        OPS="$2"
        shift
    ;;
    --joined-dict)
        JOINED_DICT_OPT="--joined-dictionary"
    ;;
    -h|--help)
        HELP=0
    ;;
    *)
        echo Unknown option '"'$key'"' >&2
        exit 1
    ;;
esac
shift
done

[[ -d "$EXP_DIR" ]] || mkdir $EXP_DIR

VALID_SETS=`echo $VALID_SETS | sed "s#^#$DATA_DIR/XXOPXX.#;s#,#,$DATA_DIR/XXOPXX.#g"`
TEST_SETS=`echo $TEST_SETS | sed "s#^#$DATA_DIR/XXOPXX.#;s#,#,$DATA_DIR/XXOPXX.#g"`

VALID_SET_OPT=""
TEST_SET_OPT=""
[[ -n $VALID_SETS ]] && VALID_SET_OPT="--validpref $VALID_SETS"
[[ -n $TEST_SETS ]] && TEST_SET_OPT="--testpref $TEST_SETS"


mkdir $EXP_DIR/data
python preprocess.py \
    --source-lang x \
    --target-lang y \
    --trainpref $DATA_DIR/all.15.train \
    $(echo $VALID_SET_OPT | sed "s/XXOPXX/all/g") \
    $(echo $TEST_SET_OPT | sed "s/XXOPXX/all/g") \
    --destdir $EXP_DIR/data \
    $JOINED_DICT_OPT \
    --workers $WORKERS

for d in train valid test; do
    for suf in bin idx; do
        mv $EXP_DIR/data/$d.x-y.x.$suf $EXP_DIR/data/all.15.$d.x-y.x.$suf
        mv $EXP_DIR/data/$d.x-y.y.$suf $EXP_DIR/data/all.15.$d.x-y.y.$suf
    done
done

for op in $OPS; do
    python preprocess.py \
        --source-lang x \
        --target-lang y \
        --trainpref $DATA_DIR/$op.15.train \
        $(echo $VALID_SET_OPT | sed "s/XXOPXX/$op/g") \
        $(echo $TEST_SET_OPT | sed "s/XXOPXX/$op/g") \
        --destdir $EXP_DIR/data \
        --srcdict $EXP_DIR/data/dict.x.txt \
        --tgtdict $EXP_DIR/data/dict.y.txt \
        --workers $WORKERS
    for d in train valid test; do
        for suf in bin idx; do
            mv $EXP_DIR/data/$d.x-y.x.$suf $EXP_DIR/data/$op.15.$d.x-y.x.$suf
            mv $EXP_DIR/data/$d.x-y.y.$suf $EXP_DIR/data/$op.15.$d.x-y.y.$suf
        done
    done
done
