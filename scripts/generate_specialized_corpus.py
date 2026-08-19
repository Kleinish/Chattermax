#!/usr/bin/env python3
import random
from datetime import date, timedelta
from pathlib import Path

ROOT=Path('/workspace') if Path('/workspace').exists() else Path(__file__).resolve().parents[1]
OUT=ROOT/'corpus_sources'; OUT.mkdir(exist_ok=True)
r=random.Random(20260814)
ones=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
tens=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
def words(n):
    if n<20:return ones[n]
    if n<100:return tens[n//10]+((' '+ones[n%10]) if n%10 else '')
    if n<1000:return ones[n//100]+' hundred'+((' '+words(n%100)) if n%100 else '')
    if n<1000000:return words(n//1000)+' thousand'+((' '+words(n%1000)) if n%1000 else '')
    return str(n)

lines=set()
for _ in range(700):
    n=r.randint(0,999999); lines.add(f'The current count is {words(n)}.')
for _ in range(350):
    a=r.randint(0,999); b=r.randint(0,99); lines.add(f'The measured value is {words(a)} point {words(b//10)} {words(b%10)}.')
for _ in range(250):
    p=r.randint(0,100); lines.add(f'The battery level is {words(p)} percent.')
for _ in range(250):
    t=r.randint(-40,120); tw=('negative '+words(abs(t))) if t<0 else words(t); lines.add(f'The temperature is {tw} degrees.')
for _ in range(250):
    h=r.randint(1,12); m=r.choice([0,5,10,15,20,25,30,35,40,45,50,55]); suffix=r.choice(['in the morning','in the afternoon','in the evening','at night'])
    mt='o clock' if m==0 else words(m); lines.add(f'The time is {words(h)} {mt} {suffix}.')
start=date(2000,1,1)
for _ in range(250):
    d=start+timedelta(days=r.randint(0,15000)); lines.add(f'The date is {d.strftime("%B")} {words(d.day)}, {words(d.year//100)} {words(d.year%100)}.')
tech_templates=[
 'The server response time is {n} milliseconds.','The network link is running at {n} megabits per second.',
 'The storage system has {n} gigabytes available.','The processor utilization is {p} percent.',
 'The device address is ten dot {a} dot {b} dot {c}.','The backup completed in {n} seconds.',
 'The sensor voltage is {d} point {e} volts.','The wireless signal is negative {n} decibels.',
 'The file contains {n} kilobytes of data.','The service will retry in {n} seconds.'
]
for _ in range(700):
    tpl=r.choice(tech_templates); vals=dict(n=words(r.randint(1,999)),p=words(r.randint(1,100)),a=words(r.randint(0,255)),b=words(r.randint(0,255)),c=words(r.randint(0,255)),d=words(r.randint(0,48)),e=words(r.randint(0,9)))
    lines.add(tpl.format(**vals))
(OUT/'numbers.txt').write_text('\n'.join(sorted(lines))+'\n',encoding='utf-8')

subjects=['The system','The service','The device','The network','The application','The sensor','The controller','The computer','The connection','The process','The request','The operation']
actions=['is ready','is available','is responding normally','has finished','has started','has been updated','has been restored','is waiting for input','requires attention','is temporarily unavailable','completed successfully','did not complete']
contexts=['right now','at the moment','this morning','this afternoon','this evening','as expected','without any errors','after the last check','before continuing','when requested','for the next step','on the first attempt']
questions=['Would you like me to continue','Should I try that again','Do you want more information','Would you like another result','Should I check the settings','Do you want to continue with this option','Would you like me to repeat that','Should I open the next item']
responses=['I found the information you requested','I can continue when you are ready','There are several possible choices','Everything appears to be working normally','No additional action is required','I will use the updated settings','The latest result is now available','The previous operation was successful']
out=set()
for s in subjects:
    for a in actions:
        out.add(f'{s} {a}.')
        for c in r.sample(contexts, min(6,len(contexts))): out.add(f'{s} {a} {c}.')
for q in questions:
    out.add(q+'?')
    for c in contexts: out.add(f'{q} {c}?')
for x in responses:
    out.add(x+'.')
    for c in contexts: out.add(f'{x} {c}.')
# Add natural short utterances and commands.
short=['Good morning.','Good afternoon.','Good evening.','Welcome home.','Please try again.','One moment please.','That is complete.','The answer is yes.','The answer is no.','I am ready.','Please continue.','Thank you.','You are welcome.','Check the next item.','Open the settings.','Read the latest message.','Start the process.','Stop the process.','Save the changes.','Cancel the request.']
out.update(short)
(OUT/'conversational.txt').write_text('\n'.join(sorted(out))+'\n',encoding='utf-8')
custom=OUT/'custom.txt'
if not custom.exists(): custom.write_text('# Add one application-specific phrase per line. Lines beginning with # are ignored.\n',encoding='utf-8')
print(f'numbers.txt: {len(lines):,} phrases')
print(f'conversational.txt: {len(out):,} phrases')
