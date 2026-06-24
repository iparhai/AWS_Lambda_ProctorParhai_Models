#!/bin/bash
# lambda_entry_script.sh
# Starts the Lambda RIE locally, or runs the handler directly on AWS.

if [ -f "/usr/local/bin/aws-lambda-rie" ] && [ "$AWS_LAMBDA_RUNTIME_API" = "" ]; then
    exec /usr/local/bin/aws-lambda-rie /usr/local/bin/python -m awslambdaric "$1"
else
    exec /usr/local/bin/python -m awslambdaric "$1"
fi
