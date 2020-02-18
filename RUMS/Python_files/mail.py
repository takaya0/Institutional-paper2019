import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

FROM_ADDRESS = 'ytk.koizumi@gmail.com'
MY_PASSWORD = 'math0714'
TO_ADDRESS = 'ra0083hr@ed.ritsumi.ac.jp'
BCC = 'receiver2@test.net'
SUBJECT = 'GmailのSMTPサーバ経由'
BODY = 'pythonでメール送信(test)'


def create_message(from_addr, to_addr, subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    #msg['Bcc'] = bcc_addrs
    msg['Date'] = formatdate()
    return msg


def send(from_addr, to_addrs, msg):
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.ehlo()
    smtpobj.starttls()
    smtpobj.ehlo()
    smtpobj.login(FROM_ADDRESS, MY_PASSWORD)
    smtpobj.sendmail(from_addr, to_addrs, msg.as_string())
    smtpobj.close()


if __name__ == '__main__':

    to_addr = TO_ADDRESS
    subject = SUBJECT
    body = BODY

    msg = create_message(FROM_ADDRESS, to_addr, subject, body)
    send(FROM_ADDRESS, to_addr, msg)
