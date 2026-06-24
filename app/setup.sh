#!/bin/bash

# Configurable variables
AWS_REGION="us-east-1"
ACCOUNT_ID="334898805711"
REPO_NAME="proctorparhai/face-proctoring"
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME"
LAMBDA_NAME="face-proctoring-lambda"

# Optional: Tag your image
IMAGE_TAG="latest"
FULL_IMAGE_URI="$ECR_URI:$IMAGE_TAG"

echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin 334898805711.dkr.ecr.us-east-1.amazonaws.com

if [ $? -ne 0 ]; then
    echo "❌ Docker login failed!"
    exit 1
fi

echo "📦 Building Docker image..."
docker build -t $FULL_IMAGE_URI .

echo "🚀 Pushing image to ECR..."
docker push $FULL_IMAGE_URI

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed!"
    exit 1
fi

echo "🔄 Updating Lambda function..."
aws lambda update-function-code \
  --function-name $LAMBDA_NAME \
  --image-uri $FULL_IMAGE_URI \
  --region $AWS_REGION

if [ $? -eq 0 ]; then
    echo "✅ Lambda function '$LAMBDA_NAME' updated successfully with image '$FULL_IMAGE_URI'."
else
    echo "❌ Lambda update failed!"
    exit 1
fi