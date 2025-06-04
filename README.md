# Setup

## With Podman

### Dependencies
1. Podman with nvidia toolkit enabled
2. Python
3. Access to eluv-io repo via ssh key

#### Download caption model
`python download_weights.py`

#### Add ssh keys to ssh-agent
`ssh-add` (on personal machine)

**NOTE**: if you are on a remote server, either you should have your ssh key on the remote server and run `ssh-add` there, or you should run it on your personal machine and verify that you are connected with agent forwarding enabled.

#### Build image
`./build.sh`

#### Default run
```
podman run --rm  --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/models:/elv/models:ro --volume=$(pwd)/tags:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=0 caption test/1.mp4 test/2.mp4
```

1. Note: you must mount the files to tag into the container storage (`--volume=$(pwd)/test:/elv/test`)
2. Tag files will appear in the `tags` directory (`--volume=$(pwd)/tags:/elv/tags`). 

#### Custom run

##### Option 1: change default runtime config
Edit the `runtime/default` section in `config.yml`

##### Option 2: pass in custom runtime config as json
```
podman run --rm  --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/models:/elv/models:ro --volume=$(pwd)/tags:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=0 caption test/1.mp4 test/2.mp4 --config '{"fps": 2}'
```