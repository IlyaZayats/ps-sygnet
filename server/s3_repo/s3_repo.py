import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os

class S3Adapter:
    def __init__(self, aws_access_key: str, aws_secret_key: str, region: str = "us-east-1", endpoint_url: str = None):
        self.session = boto3.session.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        self.s3 = self.session.client("s3", endpoint_url=endpoint_url)

    def upload_fileobj(self, file_obj, bucket_name: str, object_key: str, content_type: str = "application/octet-stream") -> bool:
        try:
            self.s3.upload_fileobj(
                Fileobj=file_obj,
                Bucket=bucket_name,
                Key=object_key,
                ExtraArgs={"ContentType": content_type}
            )
            print(f"Uploaded object to s3://{bucket_name}/{object_key}")
            return True
        except NoCredentialsError:
            print("AWS credentials not found.")
        except ClientError as e:
            print(f"Client error: {e}")
        return False