import tkinter as tk
from tkinter import messagebox, ttk
import mysql.connector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import schedule
import time
import threading

# Connect to MySQL Database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="2611#Hp1802",
    database="personal_finance",
    port=3308  
)
cursor = conn.cursor()

class FinanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personal Finance Tracker")
        
        # Set the window size to full screen
        self.root.geometry("1024x768")  
        
        self.user_id = None
        self.create_login_window()
        
        # Schedule upcoming payments check daily
        self.schedule_upcoming_payments_check()
    
    def create_login_window(self):
        # Login Window
        login_window = tk.Toplevel(self.root)
        login_window.title("Login")
        
        # Center the login window
        login_window.geometry("300x200")
        
        # Username and Password Fields
        tk.Label(login_window, text="Username").pack(pady=5)
        self.username_entry = tk.Entry(login_window)
        self.username_entry.pack(pady=5)

        tk.Label(login_window, text="Password").pack(pady=5)
        self.password_entry = tk.Entry(login_window, show='*')
        self.password_entry.pack(pady=5)
        
        # Buttons
        tk.Button(login_window, text="Login", command=self.login_user).pack(pady=5)
        tk.Button(login_window, text="Register", command=self.register_user).pack(pady=5)
    
    def login_user(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        cursor.execute("SELECT user_id FROM users WHERE username = %s AND password = %s", (username, password))
        result = cursor.fetchone()
        if result:
            self.user_id = result[0]
            self.create_main_interface()
        else:
            messagebox.showerror("Login Failed", "Invalid credentials.")
    
    def register_user(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            messagebox.showinfo("Success", "User registered successfully!")
        except mysql.connector.IntegrityError:
            messagebox.showerror("Error", "Username already exists.")
    
    def create_main_interface(self):
        # Main Interface
        self.title_label = tk.Label(self.root, text="Personal Finance Tracker", font=("Arial", 18))
        self.title_label.pack(pady=20)
        
        self.budget_button = tk.Button(self.root, text="Manage Budgets", command=self.open_budget_window)
        self.budget_button.pack(pady=10)
        
        self.expense_button = tk.Button(self.root, text="Add Expense", command=self.open_expense_window)
        self.expense_button.pack(pady=10)
        
        self.report_button = tk.Button(self.root, text="View Reports", command=self.view_reports)
        self.report_button.pack(pady=10)

        self.schedule_payment_button = tk.Button(self.root, text="Schedule Payment", command=self.open_payment_window)
        self.schedule_payment_button.pack(pady=10)

        self.manage_payments_button = tk.Button(self.root, text="Manage Scheduled Payments", command=self.manage_scheduled_payments)
        self.manage_payments_button.pack(pady=10)

        self.upcoming_payments_button = tk.Button(self.root, text="Check Upcoming Payments", command=self.check_upcoming_payments)
        self.upcoming_payments_button.pack(pady=10)

        
    def open_budget_window(self):
        budget_window = tk.Toplevel(self.root)
        budget_window.title("Manage Budgets")
        budget_window.geometry("400x300")
        
        tk.Label(budget_window, text="Category").pack(pady=5)
        self.category_entry = tk.Entry(budget_window)
        self.category_entry.pack(pady=5)
        
        tk.Label(budget_window, text="Budget Amount").pack(pady=5)
        self.budget_entry = tk.Entry(budget_window)
        self.budget_entry.pack(pady=5)
        
        tk.Button(budget_window, text="Set Budget", command=self.add_budget).pack(pady=10)
        
    def add_budget(self):
        category = self.category_entry.get()
        budget = float(self.budget_entry.get())
        
        cursor.execute("INSERT INTO categories (category_name, budget_amount) VALUES (%s, %s) ON DUPLICATE KEY UPDATE budget_amount = %s", 
                       (category, budget, budget))
        conn.commit()
        messagebox.showinfo("Success", "Budget set successfully!")
    
    def open_expense_window(self):
        expense_window = tk.Toplevel(self.root)
        expense_window.title("Add Expense")
        expense_window.geometry("400x300")
        
        cursor.execute("SELECT category_name FROM categories")
        categories = [row[0] for row in cursor.fetchall()]
        
        tk.Label(expense_window, text="Category").pack(pady=5)
        self.category_var = tk.StringVar()
        self.category_dropdown = ttk.Combobox(expense_window, textvariable=self.category_var, values=categories)
        self.category_dropdown.pack(pady=5)
        
        tk.Label(expense_window, text="Amount").pack(pady=5)
        self.expense_entry = tk.Entry(expense_window)
        self.expense_entry.pack(pady=5)
        
        tk.Label(expense_window, text="Description").pack(pady=5)
        self.description_entry = tk.Entry(expense_window)
        self.description_entry.pack(pady=5)
        
        tk.Button(expense_window, text="Add Expense", command=self.add_expense).pack(pady=10)
    
    def add_expense(self):
        category = self.category_var.get()
        amount = float(self.expense_entry.get())
        description = self.description_entry.get()
        date = datetime.now().date()
        
        cursor.execute("SELECT category_id FROM categories WHERE category_name = %s", (category,))
        category_id = cursor.fetchone()[0]
        
        cursor.execute("INSERT INTO expenses (user_id, category_id, amount, description, date) VALUES (%s, %s, %s, %s, %s)", 
                       (self.user_id, category_id, amount, description, date))
        conn.commit()
        messagebox.showinfo("Success", "Expense added successfully!")
    
    def view_reports(self):
        # Fetching category data for pie chart
        cursor.execute("SELECT c.category_name, SUM(e.amount) FROM expenses e JOIN categories c ON e.category_id = c.category_id WHERE e.user_id = %s GROUP BY e.category_id", 
                       (self.user_id,))
        data = cursor.fetchall()
        
        if data:
            categories, amounts = zip(*data)
            fig, ax = plt.subplots()
            ax.pie(amounts, labels=categories, autopct='%1.1f%%')
            ax.set_title("Spending by Category")
            
            chart_window = tk.Toplevel(self.root)
            chart_window.title("Spending Report")
            chart_canvas = FigureCanvasTkAgg(fig, master=chart_window)
            chart_canvas.draw()
            chart_canvas.get_tk_widget().pack()
        
        # Budget Progress Tracking
        cursor.execute("SELECT c.category_name, c.budget_amount, IFNULL(SUM(e.amount), 0) FROM categories c LEFT JOIN expenses e ON c.category_id = e.category_id WHERE c.budget_amount > 0 GROUP BY c.category_id")
        budget_data = cursor.fetchall()
        
        if budget_data:
            categories, budgets, expenses = zip(*budget_data)
            fig, ax = plt.subplots()
            x = range(len(categories))
            ax.bar(x, budgets, label="Budget", color="green", alpha=0.7)
            ax.bar(x, expenses, label="Spent", color="red", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45)
            ax.set_title("Budget vs Actual Spending")
            ax.legend()
            
            bar_chart_canvas = FigureCanvasTkAgg(fig, master=chart_window)
            bar_chart_canvas.draw()
            bar_chart_canvas.get_tk_widget().pack()

    def check_upcoming_payments(self):
        today = datetime.now().date()
        upcoming_date = today + timedelta(days=7)

        # Check for overdue payments (due date has passed)
        cursor.execute("SELECT description, amount, due_date FROM scheduled_payments WHERE user_id = %s AND due_date < %s", 
                       (self.user_id, today))
        overdue_payments = cursor.fetchall()
        
        # Check for upcoming payments (due date within the next 7 days)
        cursor.execute("SELECT description, amount, due_date FROM scheduled_payments WHERE user_id = %s AND due_date BETWEEN %s AND %s", 
                       (self.user_id, today, upcoming_date))
        upcoming_payments = cursor.fetchall()
        
        message = ""
        if overdue_payments:
            message += "Overdue Payments:\n"
            message += "\n".join([f"{desc}: ${amt} was due on {due}" for desc, amt, due in overdue_payments])
            message += "\n\n"
        
        if upcoming_payments:
            message += "Upcoming Payments:\n"
            message += "\n".join([f"{desc}: ${amt} due on {due}" for desc, amt, due in upcoming_payments])
        
        if message:
            messagebox.showinfo("Payments Alert", message)
        else:
            messagebox.showinfo("Payments Alert", "No overdue or upcoming payments.")

    def open_payment_window(self):
        payment_window = tk.Toplevel(self.root)
        payment_window.title("Schedule Payment")
        payment_window.geometry("400x300")
        
        tk.Label(payment_window, text="Payment Description").pack(pady=5)
        self.payment_description_entry = tk.Entry(payment_window)
        self.payment_description_entry.pack(pady=5)
        
        tk.Label(payment_window, text="Amount").pack(pady=5)
        self.payment_amount_entry = tk.Entry(payment_window)
        self.payment_amount_entry.pack(pady=5)
        
        tk.Label(payment_window, text="Due Date (YYYY-MM-DD)").pack(pady=5)
        self.payment_due_date_entry = tk.Entry(payment_window)
        self.payment_due_date_entry.pack(pady=5)
        
        tk.Button(payment_window, text="Schedule Payment", command=self.add_scheduled_payment).pack(pady=10)

    def add_scheduled_payment(self):
        description = self.payment_description_entry.get()
        amount = float(self.payment_amount_entry.get())
        due_date = self.payment_due_date_entry.get()  # User enters the due date manually

        # Check if the due date is valid
        try:
            due_date = datetime.strptime(due_date, "%Y-%m-%d").date()
        except ValueError:
            messagebox.showerror("Invalid Date", "Please enter a valid date in the format YYYY-MM-DD.")
            return
        
        cursor.execute("INSERT INTO scheduled_payments (user_id, description, amount, due_date) VALUES (%s, %s, %s, %s)", 
                       (self.user_id, description, amount, due_date))
        conn.commit()
        messagebox.showinfo("Success", "Payment scheduled successfully!")

    def manage_scheduled_payments(self):
        manage_window = tk.Toplevel(self.root)
        manage_window.title("Manage Scheduled Payments")
        manage_window.geometry("600x400")
        
        # Fetch the user's scheduled payments
        cursor.execute("SELECT payment_id, description, amount, due_date FROM scheduled_payments WHERE user_id = %s", (self.user_id,))
        payments = cursor.fetchall()
        
        # Table-like structure using Treeview
        columns = ("ID", "Description", "Amount", "Due Date")
        payment_table = ttk.Treeview(manage_window, columns=columns, show="headings")
        payment_table.heading("ID", text="ID")
        payment_table.heading("Description", text="Description")
        payment_table.heading("Amount", text="Amount")
        payment_table.heading("Due Date", text="Due Date")
        payment_table.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Populate the table
        for payment in payments:
            payment_table.insert("", tk.END, values=payment)
        
        # Delete Payment Button
        def delete_payment():
            selected_item = payment_table.selection()
            if not selected_item:
                messagebox.showerror("Error", "Please select a payment to delete.")
                return
            
            payment_id = payment_table.item(selected_item, "values")[0]  
            cursor.execute("DELETE FROM scheduled_payments WHERE payment_id = %s", (payment_id,))
            conn.commit()
            payment_table.delete(selected_item)
            messagebox.showinfo("Success", "Scheduled payment deleted successfully!")
        
        tk.Button(manage_window, text="Delete Payment", command=delete_payment).pack(pady=10)



    def schedule_upcoming_payments_check(self):
        # Schedule the payment check function to run daily
        schedule.every().day.at("10:00").do(self.check_upcoming_payments)
        
        # Start a separate thread to run the scheduler
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_schedule, daemon=True)
        scheduler_thread.start()

# Main Tkinter loop
root = tk.Tk()
app = FinanceApp(root)
root.mainloop()


     
  
       
