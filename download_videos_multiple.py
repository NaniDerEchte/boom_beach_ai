# download_videos.py
import os
import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_video(url, output_path='videos'):
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'cookiefile': '/home/nani/boom_beach_ai/cookies.txt'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def download_videos_parallel(urls, max_workers=4, output_path='videos'):
    os.makedirs(output_path, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_video, url, output_path) for url in urls]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    urls = [
        "https://www.youtube.com/live/nhC-kFHnQOU?si",
        "https://www.youtube.com/watch?v=WewjA-KjEXk",
        "https://www.youtube.com/watch?v=QKBpi26ws3g",
        "https://www.youtube.com/watch?v=Ur6mcQId8rA",
        "https://www.youtube.com/watch?v=Ry6CdIiIvu8",
        "https://www.youtube.com/watch?v=FDrzcO4ba8I",
        "https://www.youtube.com/watch?v=0gifTpSp6Ok",
        "https://www.youtube.com/watch?v=lUcmprGCoY4",
        "https://www.youtube.com/watch?v=AGlgSkYAXJ8",
        "https://www.youtube.com/watch?v=JGwJyx0_NXM",
        "https://www.youtube.com/watch?v=E9A-rGHWEeM&t",
        "https://www.youtube.com/watch?v=jiAFafQrkvk",
        "https://www.youtube.com/watch?v=SDlPzj6ypUs",
        "https://www.youtube.com/watch?v=7840o3tIGWk",
        "https://www.youtube.com/watch?v=DAOsu8n-BYE",
        "https://www.youtube.com/watch?v=0g4pDqMPpjY",
        "https://www.youtube.com/watch?v=6J_PaBznlok",
        "https://www.youtube.com/watch?v=zMYdQXWB9j0",
        "https://www.youtube.com/watch?v=7BbwPxL3cH4",
        "https://www.youtube.com/watch?v=BP2PwZpDvX0",
        "https://www.youtube.com/watch?v=cF1AD0Im474",
        "https://www.youtube.com/watch?v=2RTpHgKZMx0",
        "https://www.youtube.com/watch?v=k1jmULwqPks",
        "https://www.youtube.com/watch?v=uC9D-6ZvEa4",
        "https://www.youtube.com/watch?v=_v2htKFVUas",
        "https://www.youtube.com/watch?v=wV9t8-hvRAU",
        "https://www.youtube.com/watch?v=45C4tMpkSso",
        "https://www.youtube.com/watch?v=NUGdzFa6_Pg",
        "https://www.youtube.com/watch?v=sOwA6fERN4w",
        "https://www.youtube.com/watch?v=cciFVZFU9uQ",
        "https://www.youtube.com/watch?v=X5QOkdtfLrc",
        "https://www.youtube.com/watch?v=GQjOWv59VcE",
        "https://www.youtube.com/watch?v=gthbuoOlRm4",
        "https://www.youtube.com/watch?v=ot-vIGbFh2Q",
        "https://www.youtube.com/watch?v=ICPfcu-jNp8",
        "https://www.youtube.com/watch?v=xI_aCQSOiw8",
        "https://www.youtube.com/watch?v=2msSgMKQhpk"
    ]
    download_videos_parallel(urls, max_workers=4)
