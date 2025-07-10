from bs4 import BeautifulSoup
from flask import jsonify
from psycopg2.extras import DictCursor

import os
import psycopg2
import re


BASE_URL = "https://videos.breathecam.org/labeling"
BASE_DIR = "/workspace/projects/videos.breathecam.org/www/vids/labeling"

CLASSIFICATION_OPTIONS = [
    "smoke",
    "steam",
    "shadow",
    "other_motion",
    "none"
]


def _parse_metadata(metadata):
    video_num = int(re.search(r'Video (\d+)', metadata).group(1))
    frames_info = re.search(r'(\d+) frames starting at (\d+)', metadata)
    size_info = re.search(r'Size: (\d+)x(\d+)', metadata)
    
    if frames_info:
        start_frame = int(frames_info.group(2))
        nframes = int(frames_info.group(1))
    else:
        frames_info = re.search(r'Start Frame: (\d+)', metadata)
        start_frame = frames_info.group(1)
        size_info = re.search(r'Dimensions: (\d+)[x ](\d+)x(\d+)', metadata)
        nframes = int(size_info.group(1))
        
    view = re.search(r'View: \((\d+, \d+, \d+, \d+)\)', metadata)
    bbox = {}

    if view:
        l, t, r, b = list(map(int, view.group(1).split(", ")))
        
        bbox = {
            "left": l,
            "top": t,
            "right": r,
            "bottom": b,
            "top_left": [t, l],
            "bottom_right": [b, r],
            "height": b - t,
            "width": r - l,
        }

    return {
        "index": video_num,
        "start_frame": start_frame,
        "nframes": nframes,
        "bounding_box": bbox
    }


class AutoFEDDatabase:
    def __init__(self):
        self.conn = psycopg2.connect(dbname="smoke_detect")
        self.conn.autocommit = True

    def all_videos_for_run(self, table: str, run: str):
        """Get all videos for a specific run from the database"""
        videos = []
        classifications = {}
        
        cursor = self.conn.cursor(cursor_factory=DictCursor)
        
        try:
            cursor.execute(f"SELECT * FROM {table} WHERE run_name = %s ORDER BY id", (run,))
            
            for row in cursor.fetchall():
                classifications = row['classifications']
                metadata = row['metadata']
                video_data = _parse_metadata(metadata)       
                video_data["src"] = row["video_url"]
                video_data["record_id"] = row["id"]            
            
                for opt in CLASSIFICATION_OPTIONS:
                    if opt not in classifications:
                        classifications[opt] = False
                
                video_data["classifications"] = classifications        	           

                videos.append(video_data)
        except Exception as e:
            print(f"Database error: {e}")
            
        finally:
            cursor.close()
        
        videos.sort(key=lambda v: v["index"])
        
        return videos

    def available_runs(self, table: str):
        runs = []
    
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(f"SELECT DISTINCT run_name FROM {table} ORDER BY run_name")
            runs = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            cursor.close()
            
        return runs
    
    def close(self):
        self.conn.close()

    def create_table(self, table: str):
        cursor = self.conn.cursor()

        cursor.execute(f"""
CREATE TABLE IF NOT EXISTS {table} (
    id SERIAL PRIMARY KEY,
    run_name TEXT NOT NULL,
    video_url TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
        self.conn.commit()

        cursor.close()
        
    def delete_run(self, table: str, run: str):
        cursor = self.conn.cursor()

        cursor.execute("DELETE FROM video_labels WHERE run_name = %s", (run,))

        self.conn.commit()

        cursor.close()
        
        print(f"Deleted existing records for run_name: {run}")

    def drop_table(self, table: str):
        cursor = self.conn.cursor()

        cursor.execute(f"DROP TABLE IF EXISTS {table}")

        self.conn.commit()

        cursor.close()

    def insert(self, table: str, results):
        cursor = self.conn.cursor()

        for item in results:
            cursor.execute(
                f"INSERT INTO {table} (run_name, video_url, metadata) VALUES (%s, %s, %s)",
                (item['run_name'], item['video_src'], item['metadata'])
            )

        self.conn.commit()

        cursor.close()

        print(f"Successfully inserted {len(results)} video records into the database.")

    def insert_from_html(self, table: str, run: str, html_file: str):
        soup = BeautifulSoup(open(html_file, "r").read())
        video_containers = soup.find_all("div", class_="video-container")

        results = []

        for container in video_containers:  # Process all videos
            video_src = os.path.join(self.base_directory, container.find('video')['src'])
            video_src = os.path.realpath(video_src)
            
            assert video_src.startswith(self.base_directory)
            
            video_src = f"{BASE_URL}/{video_src[len(self.base_directory):]}"
            label = container.find('div', class_='label')

            for br in label.find_all('br'):
                br.replace_with(' ')

            metadata = label.text.strip()
            
            results.append({
                'run_name': os.path.splitext(os.path.basename(html_file))[0],
                'video_src': video_src,
                'metadata': metadata
            })
            
            self.insert(table, run, results)
    
    def labeled_videos_for_run(self, run_name, classification_filter=None):
        """Get all videos for a specific run from the database"""
        videos = []
        classifications = {}
        cur = self.conn.cursor(cursor_factory=DictCursor)
        
        classification_filter = classification_filter or (lambda c: len(c) > 0) 
        try:
            cur.execute("SELECT * FROM video_labels WHERE run_name = %s ORDER BY id", (run_name,))
            
            for row in cur.fetchall():
                classifications = row['classifications']

                if not classification_filter(classifications):
                    continue	

                metadata = row['metadata']
                video_data = _parse_metadata(metadata)
                video_data["src"] = row["video_url"]
                video_data["record_id"] = row["id"]
                video_data["classifications"] = [k for k in classifications.keys() if classifications.get(k, False)]

                videos.append(video_data)
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            cur.close()
        
        videos.sort(key=lambda v: v["index"])
        
        return videos
    
    def set_classifications(self, video_id, classifications):
        try:
            cursor = self.conn.cursor(cursor_factory=DictCursor)

            cursor.execute( "UPDATE video_labels SET classifications = %s WHERE id = %s", (classifications, video_id))

            self.conn.commit()
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'message': f'Error: {str(e)}'}), 500
        finally:
            cursor.close()

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.conn.close()
