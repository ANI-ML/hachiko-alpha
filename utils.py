import os
from dotenv import load_dotenv
from pathlib import Path

import psutil
import datetime
import gc # garbage collection
import boto3
from datetime import datetime
import csv
import json
from graph import create_graph
from memory_utils import get_memory_log


env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

class OutputTracker:
    def __init__(self):
        # Check if AWS credentials are set
        print(f"AWS Access Key exists: {bool(os.getenv('AWS_ACCESS'))}")
        print(f"AWS Secret exists: {bool(os.getenv('AWS_SECRET'))}")

         # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS"),
            aws_secret_access_key = os.getenv("AWS_SECRET")
            )
        self.bucket_name = 'ani.ml-hachiko-testing'
        self.performance_log = "performance_logs/performance_tracking.csv"

        if not self.verify_bucket_access():
            raise Exception("Failed to verify S3 bucket access")

    def verify_bucket_access(self):
        """Verify S3 bucket access and permissions by trying to write to the performance log"""
        try:
            # Check if we can write to the performance log
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.performance_log,
                Body=""
            )
            print("Write permissions verified")
            return True
        except Exception as e:
            print(f"Bucket access verification failed: {str(e)}")
            return False

    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = {
            'memory_used_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'system_memory': {
                'total': psutil.virtual_memory().total / 1024 / 1024,
                'available': psutil.virtual_memory().available / 1024 / 1024,
                'used_percent': psutil.virtual_memory().percent
            }
        }
        return memory_info
        
    def save_to_s3(self, file_path, s3_path):
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_path)
            return True
        except Exception as e:
            print(f"Error uploading to S3: {str(e)}")
            traceback.print_exc()
            return False
            
    def log_run(self, uploaded_files, generated_summary, performance_metrics):
        """Log a single run with multiple PDFs and one summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        memory_stats = self.get_memory_usage()

        try:
            # Create metadata
            metadata = {
                'timestamp': timestamp,
                'model_version': 'v1.1',
                'processing_time': performance_metrics.get('processing_time'),
                'input_filename': [file.name for file in uploaded_files],
                'number_of_files': len(uploaded_files),
                'summary_length': len(generated_summary),
                'success': performance_metrics.get('success', True),
                'memory_usage': {
                    'process_memory_mb': memory_stats['memory_used_mb'],
                    'process_memory_percent': memory_stats['memory_percent'],
                    'system_memory_total_mb': memory_stats['system_memory']['total'],
                    'system_memory_available_mb': memory_stats['system_memory']['available'],
                    'system_memory_used_percent': memory_stats['system_memory']['used_percent']
                },
                'memory_log': get_memory_log()
            }
            
            # Save PDF(s) to S3
            print(f"Attempting to save {len(uploaded_files)} files")
            pdf_paths = []
            for pdf_file in uploaded_files:
                print(f"Processing file: {pdf_file.name}")
                pdf_path = f"pdfs/{run_id}/{pdf_file.name}"
                try:
                    pdf_file.seek(0)
                    self.s3_client.upload_fileobj(
                        pdf_file,
                        self.bucket_name,
                        pdf_path
                    )
                    pdf_paths.append(pdf_path)
                    print(f"Successfully uploaded {pdf_file.name}")
                except Exception as e:
                    print(f"Failed to upload {pdf_file.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Save summary to S3
            print("Attempting to save summary")
            summary_path = f"summaries/{run_id}/summary.txt"
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=summary_path,
                    Body=generated_summary
                )
                print("Successfully saved summary")
            except Exception as e:
                print(f"Failed to save summary: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Save metadata to S3
            print("Attempting to save metadata")
            metadata_path = f"metadata/{run_id}/metadata.json"
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=metadata_path,
                    Body=json.dumps(metadata, indent=4)
                )
                print("Successfully saved metadata")
            except Exception as e:
                print(f"Failed to save metadata: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Update performance log
            try:
                print("Attempting to update performance log")
                self.update_performance_log(metadata)
                print("Successfully updated performance log")
            except Exception as e:
                print(f"Failed to update performance log: {str(e)}")
                import traceback
                traceback.print_exc()
            
            return {
                'run_id': run_id,
                'pdf_path': pdf_paths,
                'summary_path': summary_path,
                'metadata_path': metadata_path
            }
            
        except Exception as e:
            print(f"Error in logging run: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def update_performance_log(self, metadata):
        """Update the running performance log"""
        try:
            # Try to get existing log
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=self.performance_log
                )
                existing_log = response['Body'].read().decode('utf-8')
            except:
                existing_log = "timestamp,filenames,num_files,processing_time,summary_length,success,memory_used_mb,memory_percent,system_memory_used_percent\n"
            
            # Add new entry
            filenames = "|".join(metadata['input_filename'])
            new_entry = (
                f"{metadata['timestamp']},"
                f"{filenames},"
                f"{metadata['number_of_files']},"
                f"{metadata['processing_time']},"
                f"{metadata['summary_length']},"
                f"{metadata['success']},"
                f"{metadata['memory_usage']['process_memory_mb']:.2f},"
                f"{metadata['memory_usage']['process_memory_percent']:.2f},"
                f"{metadata['memory_usage']['system_memory_used_percent']:.2f}\n"
            )
            updated_log = existing_log + new_entry
            
            # Save updated log
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.performance_log,
                Body=updated_log
            )
            
        except Exception as e:
             print(f"Error updating performance log: {str(e)}")
 

# try:
#     tracker = OutputTracker()
#     # This will immediately test if your AWS credentials work
#     # and if you have access to the bucket
# except Exception as e:
#     print(f"Failed to initialize OutputTracker: {e}")