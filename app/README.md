### Deployment [AWS]:
```bash
#!/bin/bash

TAG=${1:-latest}
ECR_REPO="334898805711.dkr.ecr.us-east-1.amazonaws.com/proctorparhai/compiler"

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO
docker build -t $ECR_REPO:$TAG .
docker push $ECR_REPO:$TAG

echo "Pushed: $ECR_REPO:$TAG"
```

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 334898805711.dkr.ecr.us-east-1.amazonaws.com