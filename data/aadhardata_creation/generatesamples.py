import csv
import random
from datetime import datetime, timedelta
 
first_names = [
    "Amit", "Priya", "Rahul", "Sneha", "Vijay", "Neha", "Arjun", "Meena",
    "Rakesh", "Nisha", "Anil", "Kavya", "Suresh", "Divya", "Ravi", "Pooja",
    "Manoj", "Tanya", "Deepak", "Jyoti", "Karan", "Sunita", "Siddharth", "Rekha",
    "Ajay", "Simran", "Varun", "Anjali", "Nitin", "Isha", "Abhishek", "Kiran",
    "Mohit", "Ayesha", "Sanjay", "Preeti", "Rohit", "Lavanya", "Yogesh", "Asmita",
    "Harish", "Ruchi", "Rajeev", "Bhavna", "Sameer", "Madhu", "Gaurav", "Leena",
    "Vinay", "Pallavi", "Naveen", "Harsha", "Aniket", "Nandini", "Kunal", "Sheetal",
    "Parth", "Snehal", "Raj", "Aparna", "Hemant", "Deepti", "Sahil", "Ritika",
    "Uday", "Mitali", "Sharad", "Neelam", "Tushar", "Juhi", "Tarun", "Shalini",
    "Dev", "Naina", "Bhavesh", "Charu", "Nikhil", "Vandana", "Dinesh", "Gayatri",
    "Akhil", "Komal", "Om", "Shreya", "Rajesh", "Tanvi", "Chirag", "Ipsita",
    "Lakshya", "Trisha", "Hardik", "Sanya", "Keshav", "Amrita", "Aarav", "Madhavi",
    "Aditya", "Diya", "Aryan", "Kritika"
]
 
last_names = [
    "Sharma", "Verma", "Kumar", "Yadav", "Kapoor", "Singh", "Joshi", "Mehta",
    "Chopra", "Bhatia", "Das", "Patel", "Reddy", "Rana", "Naidu", "Ghosh",
    "Agarwal", "Iyer", "Mishra", "Tiwari", "Pandey", "Bhatt", "Gupta", "Malhotra",
    "Dube", "Shetty", "Kulkarni", "Menon", "Pillai", "Chatterjee", "Roy", "Dutta",
    "Bose", "Nair", "Rao", "Saxena", "Sinha", "Tripathi", "Thakur", "Khatri",
    "Rawat", "Gill", "Sandhu", "Kohli", "Ahluwalia", "Vohra", "Kalra", "Chhabra",
    "Grover", "Puri", "Vyas", "Ojha", "Tyagi", "Sood", "Saraf", "Pande", "Bhagat",
    "Desai", "Bhonsle", "Bagga", "Mahajan", "Barot", "Dubey", "Kamble", "Chouhan",
    "Bhadoria", "Khare", "Mittal", "Tandon", "Zaveri", "Jain", "Modi", "Damani",
    "Lal", "Kaushik", "Wadhwa", "Goel", "Nagpal", "Ahuja", "Sabharwal", "Bajaj",
    "Narang", "Makhija", "Vaswani", "Talwar", "Mathew", "Jacob", "Thomas", "Paul",
    "Fernandes", "Pereira", "D’Costa", "Carvalho", "D’Souza", "Menezes", "Martins",
    "Gonsalves", "Dias", "Rodrigues", "Mascarenhas"
]
 
def generate_dob():
    start_date = datetime.strptime("01/01/1980", "%d/%m/%Y")
    end_date = datetime.strptime("31/12/2006", "%d/%m/%Y")
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    dob = start_date + timedelta(days=random_days)
    return dob.strftime("%d/%m/%Y")
 
def generate_unique_numbers(count, length):
    unique_numbers = set()
    while len(unique_numbers) < count:
        number = ''.join(str(random.randint(0, 9)) for _ in range(length))
        unique_numbers.add(number)
    return list(unique_numbers)
 
def generate_data(num_records=300):
    aadhar_numbers = generate_unique_numbers(num_records, 12)
    vid_numbers = generate_unique_numbers(num_records, 16)
    data = []
 
    for i in range(num_records):
        first = random.choice(first_names)
        last = random.choice(last_names)
        name = f"{first} {last}"
        dob = generate_dob()
        aadhar = f"{aadhar_numbers[i][:4]} {aadhar_numbers[i][4:8]} {aadhar_numbers[i][8:]}"
        vid = f"{vid_numbers[i][:4]} {vid_numbers[i][4:8]} {vid_numbers[i][8:12]} {vid_numbers[i][12:]}"
        data.append([name, dob, aadhar, vid])
 
    return data
 
def save_to_csv(filename, records):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'dob', 'aadhar number', 'vid'])
        writer.writerows(records)
 
if __name__ == "__main__":
    records = generate_data(300)
    save_to_csv("people_data.csv", records)