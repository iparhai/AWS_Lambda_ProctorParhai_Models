#!/bin/sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
  exec /usr/local/bin/aws-lambda-rie /usr/local/bin/python3 -m awslambdaric "$@"
else
  exec /usr/local/bin/python3 -m awslambdaric "$@"
fi


# https://medium.com/@nicholaszolton/fastapi-on-aws-lambda-with-docker-5ca5f76a9880