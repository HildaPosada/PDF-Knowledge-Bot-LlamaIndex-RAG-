import base64, io, json, re
from http.server import BaseHTTPRequestHandler
from pypdf import PdfReader
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size<=0 or size>4*1024*1024:return self.reply(413,{'error':'Choose a PDF under 3 MB.'})
            data=json.loads(self.rfile.read(size))
            query=str(data.get('question','')).strip()
            if not query or len(query)>500:return self.reply(400,{'error':'Enter a question under 500 characters.'})
            pdf=PdfReader(io.BytesIO(base64.b64decode(data['pdf'],validate=True)))
            if pdf.is_encrypted:return self.reply(400,{'error':'Encrypted PDFs are not supported.'})
            if len(pdf.pages)>100:return self.reply(400,{'error':'Use a PDF with at most 100 pages.'})
            stop={'what','is','the','a','an','in','of','to','and','are','does','it','how','this','that'}
            terms=set(re.findall(r'[a-z0-9]+',query.lower()))-stop
            matches=[]
            for page_number,page in enumerate(pdf.pages,1):
                text=page.extract_text() or ''
                for paragraph in re.split(r'\n\s*\n|(?<=[.!?])\s+',text):
                    words=set(re.findall(r'[a-z0-9]+',paragraph.lower()))
                    score=len(terms & words)
                    if score:matches.append({'page':page_number,'text':paragraph.strip()[:2000],'score':score})
            matches.sort(key=lambda item:(-item['score'],item['page']))
            self.reply(200,{'matches':matches[:4],'pages':len(pdf.pages)})
        except Exception:
            self.reply(400,{'error':'This PDF could not be read. Upload a valid, text-based PDF.'})
    def reply(self,status,body):
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(json.dumps(body).encode())
