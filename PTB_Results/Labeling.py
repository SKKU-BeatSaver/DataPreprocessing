import os
from pathlib import Path
import re
import numpy as np
from scipy.stats import norm

class RMSSDRanges:
    """Age-specific RMSSD ranges"""
    
    @staticmethod
    def get_range(age):
        """Get RMSSD range for specific age group"""
        if 10 <= age <= 19:
            return {
                'median_low': 53.0, 'median_high': 66.5,
                'range_low': 17.4, 'range_high': 232.2
            }
        elif 20 <= age <= 29:
            return {
                'median_low': 41.0, 'median_high': 48.5,
                'range_low': 13.9, 'range_high': 161.4
            }
        elif 30 <= age <= 39:
            return {
                'median_low': 35.0, 'median_high': 37.5,
                'range_low': 11.0, 'range_high': 129.2
            }
        elif 40 <= age <= 49:
            return {
                'median_low': 27.0, 'median_high': 30.4,
                'range_low': 8.8, 'range_high': 113.7
            }
        elif 50 <= age <= 59:
            return {
                'median_low': 24.4, 'median_high': 25.6,
                'range_low': 6.9, 'range_high': 103.4
            }
        elif 60 <= age <= 69:
            return {
                'median_low': 20.4, 'median_high': 20.7,
                'range_low': 5.6, 'range_high': 104.8
            }
        elif 70 <= age <= 79:
            return {
                'median_low': 17.8, 'median_high': 17.9,
                'range_low': 4.7, 'range_high': 120.9
            }
        else:  # 80+
            return {
                'median_low': 15.6, 'median_high': 16.1,
                'range_low': 3.9, 'range_high': 158.3
            }

class ECGProcessor:
    def __init__(self, directory="."):
        self.directory = Path(directory)
        self.file_pairs = {}
        
    def scan_files(self):
        """Scan directory and group related files together"""
        pattern = re.compile(r'S(\d{4})_(.+)\.txt')
        
        for file in self.directory.glob("S*.txt"):
            match = pattern.match(file.name)
            if match:
                subject_id = match.group(1)
                file_type = match.group(2)
                
                if subject_id not in self.file_pairs:
                    self.file_pairs[subject_id] = {}
                    
                self.file_pairs[subject_id][file_type] = file
                
        return self.file_pairs
    
    def read_results_file(self, results_filename):
        """Read and extract information from results file"""
        info = {}
        
        with open(results_filename, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                if line.startswith('#Age:'):
                    info['age'] = int(line.split(':')[1].strip())
                elif line.startswith('#Sex:'):
                    info['sex'] = line.split(':')[1].strip().lower()
                elif 'RMSSD =' in line:
                    rmssd = float(re.search(r'RMSSD = ([\d.]+)', line).group(1))
                    info['rmssd'] = rmssd
                elif 'Mean HR =' in line:
                    mean_hr = float(re.search(r'Mean HR = ([\d.]+)', line).group(1))
                    info['mean_hr'] = mean_hr
                elif 'Number of R-peaks =' in line:
                    r_peaks = int(re.search(r'Number of R-peaks = (\d+)', line).group(1))
                    info['r_peaks'] = r_peaks
        
        return info
    
    def calculate_rmssd_probability(self, rmssd_value, age):
        """Calculate probability of RMSSD being in normal range"""
        # Convert RMSSD to milliseconds
        rmssd_ms = rmssd_value * 1000
        
        # Get age-specific ranges
        ranges = RMSSDRanges.get_range(age)
        
        # Calculate median for the age group
        median = (ranges['median_low'] + ranges['median_high']) / 2
        
        # Calculate range width
        range_width = ranges['range_high'] - ranges['range_low']
        
        # Calculate standard deviation (assuming range represents 96% of data)
        sd = range_width / (2 * 1.96)
        
        # Calculate z-score
        z_score = (rmssd_ms - median) / sd
        
        # Calculate probability using normal distribution
        if ranges['range_low'] <= rmssd_ms <= ranges['range_high']:
            probability = 1.0 - abs(norm.cdf(z_score) - 0.5) * 2
        else:
            if rmssd_ms < ranges['range_low']:
                distance = (ranges['range_low'] - rmssd_ms) / ranges['range_low']
            else:
                distance = (rmssd_ms - ranges['range_high']) / ranges['range_high']
            probability = np.exp(-distance)
        
        return probability, ranges
    
    def process_subject(self, subject_id):
        """Process data for a single subject"""
        if subject_id not in self.file_pairs:
            raise ValueError(f"No files found for subject {subject_id}")
            
        files = self.file_pairs[subject_id]
        
        if 'channel_0' not in files or 'results' not in files:
            raise ValueError(f"Missing required files for subject {subject_id}")
        
        # Read ECG data
        ecg_data = np.loadtxt(files['channel_0'])
        
        # Read results and extract info
        info = self.read_results_file(files['results'])
        
        # Calculate probability
        probability, ranges = self.calculate_rmssd_probability(info['rmssd'], info['age'])
        
        # Create labeled data
        labeled_data = np.append(ecg_data, probability)
        
        # Save labeled data
        output_filename = self.directory / f"S{subject_id}_labeled.txt"
        np.savetxt(
            output_filename,
            labeled_data,
            fmt='%.16e',
            delimiter='\n'
        )
        
        # Prepare results summary
        results = {
            'subject_id': subject_id,
            'info': info,
            'probability': probability,
            'ranges': ranges,
            'labeled_data': labeled_data,
            'output_file': output_filename
        }
        
        return results
    
    def process_all(self):
        """Process all file pairs found in the directory"""
        results = {}
        
        for subject_id in self.file_pairs:
            try:
                results[subject_id] = self.process_subject(subject_id)
                self.print_analysis_summary(results[subject_id])
            except Exception as e:
                print(f"Error processing subject {subject_id}: {str(e)}")
                
        return results
    
    def print_analysis_summary(self, result):
        """Print analysis summary for a subject"""
        print(f"\nAnalysis Summary for Subject {result['subject_id']}:")
        print("=" * 50)
        print(f"Age: {result['info']['age']} years")
        print(f"RMSSD: {result['info']['rmssd']*1000:.2f} ms")
        print(f"Age group normal ranges:")
        print(f"  Median: {result['ranges']['median_low']:.1f}-{result['ranges']['median_high']:.1f} ms")
        print(f"  Range: {result['ranges']['range_low']:.1f}-{result['ranges']['range_high']:.1f} ms")
        print(f"Probability of normal range: {result['probability']*100:.2f}%")
        print("=" * 50)

# Example usage
if __name__ == "__main__":
    try:
        # Initialize processor
        processor = ECGProcessor()
        
        # Scan for files
        file_pairs = processor.scan_files()
        
        # Print found files
        print("Found file pairs:")
        for subject_id, files in file_pairs.items():
            print(f"\nSubject {subject_id}:")
            for file_type, file_path in files.items():
                print(f"  {file_type}: {file_path}")
        
        # Process all files
        results = processor.process_all()
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")