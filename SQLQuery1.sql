CREATE DATABASE ATTENDACE;
use ATTENDACE;


CREATE TABLE users (
    id INT IDENTITY(1,1) PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    photo_path VARCHAR(255),
    embedding NVARCHAR(255)
);

CREATE TABLE attendance (
    id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT,
    timestamp DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

SELECT * FROM DBO.users;