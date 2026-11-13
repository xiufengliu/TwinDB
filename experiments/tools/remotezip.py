import io, sys, urllib.request, zipfile, os, ssl
URL = "https://zenodo.org/api/records/6563114/files/3_years_3021_smart_heat_meters_residential_denmark.zip/content"
class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url; self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        r = urllib.request.urlopen(req, timeout=60)
        self.size = int(r.headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def tell(self): return self.pos
    def readinto(self, b):
        n = len(b)
        if self.pos >= self.size or n == 0: return 0
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (self.pos, end)})
        data = urllib.request.urlopen(req, timeout=120).read()
        b[:len(data)] = data; self.pos += len(data); return len(data)
if __name__ == "__main__":
    f = io.BufferedReader(RangeFile(URL), buffer_size=1 << 20)
    z = zipfile.ZipFile(f)
    if sys.argv[1] == "ls":
        for i in z.infolist(): print(i.filename, i.file_size)
    else:
        out = sys.argv[2]
        pats = sys.argv[3:]
        for i in z.infolist():
            if any(p in i.filename for p in pats):
                dst = os.path.join(out, i.filename); os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(i) as src, open(dst, "wb") as d: d.write(src.read())
                print("got", i.filename, flush=True)
