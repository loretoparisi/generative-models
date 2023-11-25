#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# @2023 Loreto Parisi
#
# LP: freeze installed requirements
REQUIREMENTS=$1
pip freeze > requirements-lock.txt

# LP: split lib name without version (lib==version)
cat $REQUIREMENTS | cut -d '=' -f 1 | cut -d '>' -f 1 | sed -r 's/[-]+/_/g' | sed -r 's/protobuf/google.protobuf/g' | xargs -I {} /bin/bash -c ' echo {};'

# LP: check installed versions
cat $REQUIREMENTS | cut -d '=' -f 1 | cut -d '>' -f 1 | sed -r 's/[-]+/_/g' | sed -r 's/protobuf/google.protobuf/g' | xargs -I {} python3 -c $'import {} as lib;\nprint(lib.__name__, lib.__version__)'

# LP: pickle version
echo pickle `python3 -c "import pickle; print(pickle.format_version);"`
