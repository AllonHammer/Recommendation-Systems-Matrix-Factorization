#!/bin/bash
pushd ..
zip -r recsys.zip ./
aws s3 cp recsys.zip s3://browsi-temp/Recsys/
popd
aws s3 cp bootstrapping.sh s3://browsi-temp/Recsys/
pushd ..
rm recsys.zip