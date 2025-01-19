from flask import Flask, render_template, request, redirect, url_for
from face_recognition_module import FaceRecognition
from object_recognition_module import ObjectRecognition
from user_management_module import UserManagement
from auth_module import authenticate

app = Flask(__name__)
face_recognition = FaceRecognition()
object_recognition_instance = ObjectRecognition()
user_management = UserManagement()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form['password']
        if authenticate(password):
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Senha incorreta.')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        user_management.register_user(name)
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/list-users')
def list_users():
    users = user_management.list_users()
    return render_template('list_users.html', users=users)

@app.route('/update-user', methods=['GET', 'POST'])
def update_user():
    if request.method == 'POST':
        old_name = request.form['old_name']
        new_name = request.form['new_name']
        user_management.update_user(old_name, new_name)
        return redirect(url_for('dashboard'))
    return render_template('update_user.html')

@app.route('/delete-user', methods=['GET', 'POST'])
def delete_user():
    if request.method == 'POST':
        name = request.form['name']
        user_management.delete_user(name)
        return redirect(url_for('dashboard'))
    return render_template('delete_user.html')

@app.route('/recognize')
def recognize():
    face_recognition.live_recognition()
    return redirect(url_for('dashboard'))

@app.route('/object-recognition')
def object_recognition_route():
    object_recognition_instance.object_recognition()
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
