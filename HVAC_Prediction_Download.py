import os
import concurrent.futures
import s3fs
import time
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
dataset_year = '2024'
dataset_name = 'comstock_amy2018_release_2'
upgrade_id = 1
state = 'NY'
s3_path = f's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/timeseries_individual_buildings/by_state/upgrade={upgrade_id}/state={state}/'
local_dir = f'./downloaded_data/{dataset_year}/{dataset_name}/upgrade_{upgrade_id}/{state}/'
os.makedirs(local_dir, exist_ok=True)

# Create anonymous S3 filesystem
fs = s3fs.S3FileSystem(anon=True)

def download_file(s3_file_path):
    """Download a single file from S3 to local storage"""
    try:
        # Create local path preserving directory structure
        relative_path = s3_file_path.split(f"{state}/")[-1]
        local_file_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download file with progress tracking
        start_time = time.time()
        fs.get(s3_file_path, local_file_path)
        download_time = time.time() - start_time
        
        file_size = os.path.getsize(local_file_path) / (1024 * 1024)  # in MB
        speed = file_size / download_time if download_time > 0 else 0
        
        return {
            'status': 'success',
            'file': s3_file_path,
            'local_path': local_file_path,
            'size_mb': file_size,
            'time_sec': download_time,
            'speed_mbps': speed
        }
    except Exception as e:
        return {
            'status': 'error',
            'file': s3_file_path,
            'error': str(e)
        }

def main():
    # List all files in S3 path
    try:
        all_files = fs.ls(s3_path, detail=False)
        logging.info(f"Found {len(all_files)} files to download")
    except Exception as e:
        logging.error(f"Failed to list files: {e}")
        return

    # Create progress bar
    progress = tqdm(total=len(all_files), desc="Downloading files", unit="file")
    
    # Download files in parallel with thread pool
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Submit all download tasks
        future_to_file = {executor.submit(download_file, file): file for file in all_files}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    progress.set_postfix({
                        'speed': f"{result['speed_mbps']:.1f}MB/s",
                        'size': f"{result['size_mb']:.1f}MB"
                    })
                else:
                    logging.error(f"Failed {file}: {result['error']}")
                
            except Exception as e:
                logging.error(f"Exception for {file}: {e}")
                results.append({
                    'status': 'exception',
                    'file': file,
                    'error': str(e)
                })
            finally:
                progress.update(1)
    
    progress.close()
    
    # Generate download report
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    total_size = sum(r['size_mb'] for r in results if r['status'] == 'success')
    avg_speed = sum(r['speed_mbps'] for r in results if r['status'] == 'success') / success_count if success_count > 0 else 0
    
    logging.info(f"Download complete: {success_count} succeeded, {error_count} failed")
    logging.info(f"Total data downloaded: {total_size:.2f} MB")
    logging.info(f"Average download speed: {avg_speed:.2f} MB/s")
    
    # Save error log
    if error_count > 0:
        with open(os.path.join(local_dir, 'download_errors.log'), 'w') as f:
            f.write("Failed downloads:\n")
            for result in results:
                if result['status'] != 'success':
                    f.write(f"{result['file']}: {result['error']}\n")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds")