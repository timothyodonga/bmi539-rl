#!/bin/bash
source image_description.conf
dockerfile="Dockerfile"
override_latest="true"

function build_image() {
    local image_name=$1
    local image_tag=$2
    local dockerfile=$3
    local override_latest=$4
    [[ -z $override_latest ]] && latest_string="" || latest_string="-t $image_name:latest"
    ## pytorch base image does not have arm64, although the base nvidia image does
    # platform="linux/amd64,linux/arm64"

    platform="linux/amd64"
    echo "Building image: $image_name:$image_tag"
    docker buildx build --platform $platform --force-rm --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from=type=registry,ref=${CONTAINER_HOST_ACCOUNT}/${image_name} --push -t $image_name:$image_tag $latest_string -f $dockerfile .
}

build_image $image_name $image_tag $dockerfile $override_latest
