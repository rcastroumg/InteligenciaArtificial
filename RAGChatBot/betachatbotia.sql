CREATE TABLE conversation (
    idconversation INT AUTO_INCREMENT PRIMARY KEY,
    user VARCHAR(255) NOT NULL,
    session TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversation_history (
    idconversation INT,
    idhistory INT AUTO_INCREMENT PRIMARY KEY,
    role VARCHAR(30),
    message VARCHAR(1000),
    retrieved_docs JSON,
    FOREIGN KEY (idconversation) REFERENCES conversation(idconversation)
);