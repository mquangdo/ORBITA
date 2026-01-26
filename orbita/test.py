import imaplib
import email
import smtplib
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain.tools import tool

IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
EMAIL_ADDRESS = "dominhquang02092004@gmail.com"
APP_PASSWORD = "ndekffuzwhnobldp"

def fetch_emails_tool(k: int, target_email: str = None) -> list:
    """
    Đọc nội dung k email mới nhất. Có thể lọc theo địa chỉ người gửi.
    """
    emails_found = []
    try:
        imap = imaplib.IMAP4_SSL(IMAP_SERVER, 993)
        imap.login(EMAIL_ADDRESS, APP_PASSWORD)
        imap.select("INBOX")
        
        if target_email:
            search_query = f'FROM "{target_email}"'
        else:
            search_query = "ALL"
            
        status, messages = imap.search(None, search_query)
        
        if status == "OK" and messages[0]:
            ids = messages[0].split()[-k:][::-1]
            
            for mail_id in ids:
                _, msg_data = imap.fetch(mail_id, "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])
                
                # Giải mã tiêu đề
                subject, encoding = decode_header(msg.get("Subject", ""))[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8", errors="ignore")
                
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            payload = part.get_payload(decode=True)
                            body = payload.decode(errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")

                emails_found.append({
                    "from": msg.get("From"),
                    "subject": subject,
                    "content": body.strip(),
                    "id": mail_id.decode()
                })
        
        imap.logout()
    except Exception as e:
        return [f"Lỗi khi đọc mail: {str(e)}"]
        
    return emails_found

if __name__ == "__main__":
    recent_emails = fetch_emails_tool(1, target_email="nguyenthuytrang1_t67@hus.edu.vn")
    print(recent_emails)    