#!/bin/bash

rm -rf test_output/
mkdir test_output

##--volume=$(pwd)/models:/elv/models:ro
podman run --rm  --volume=$(pwd)/test:/elv/test:ro  --volume=$(pwd)/test_output:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=0 caption test/1.mp4 test/2.mp4

ex=$?

cd test_output
find

exit $ex
