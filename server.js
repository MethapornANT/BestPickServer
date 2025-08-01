process.removeAllListeners('warning');

const express = require("express");
const bodyParser = require("body-parser");
const mysql = require("mysql2");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const admin = require("firebase-admin");
const cors = require("cors");
const axios = require("axios");
const fs = require("fs");
const crypto = require("crypto");
const nodemailer = require("nodemailer");
const multer = require("multer");
require("dotenv").config();
const path = require("path");
const JWT_SECRET = process.env.JWT_SECRET;
const app = express();

const { PythonShell } = require('python-shell');
const promptpay = require('promptpay-qr');
const QRCode = require('qrcode');

// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors()); // Enable CORS

// Initialize Firebase Admin SDK
const serviceAccount = require("./config/apilogin-6efd6-firebase-adminsdk-b3l6z-c2e5fe541a.json");
const { title } = require("process");
const { error } = require("console");
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

// Create Connection Pool
const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port: process.env.DB_PORT,
  waitForConnections: true,
  connectionLimit: 20,
  queueLimit: 0,
  connectTimeout: 60000,
  ssl: {
    rejectUnauthorized: false, // กำหนดว่าเซิร์ฟเวอร์ต้องมีใบรับรองที่น่าเชื่อถือ
    ca: fs.readFileSync("./certs/isrgrootx1.pem"), // เพิ่มไฟล์ใบรับรอง
  },
});

// ฟังก์ชันสำหรับการเชื่อมต่อใหม่อัตโนมัติ
function reconnect() {
  pool.getConnection((err) => {
    if (err) {
      console.error("Error re-establishing database connection: ", err);
      setTimeout(reconnect, 2000); // ลองเชื่อมต่อใหม่ทุก 2 วินาที
    } else {
      console.log("Database reconnected successfully.");
    }
  });
}

// ตรวจจับข้อผิดพลาดใน Pool และเชื่อมต่อใหม่อัตโนมัติ
pool.on('error', (err) => {
  if (err.code === 'PROTOCOL_CONNECTION_LOST' || err.code === 'ECONNRESET') {
    console.error("Database connection lost. Reconnecting...");
    reconnect(); // เรียกใช้ reconnect
  } else {
    console.error("Database error: ", err);
    throw err;
  }
});

// ตรวจสอบการเชื่อมต่อเริ่มต้น
pool.getConnection((err, connection) => {
  if (err) {
    console.error("Error connecting to the database:", err);
    return;
  }
  console.log("Connected to the database successfully!");
  connection.release(); // ปล่อยการเชื่อมต่อกลับไปใน Pool
});

module.exports = pool; // Export pool เพื่อให้สามารถใช้งานในไฟล์อื่นๆ ได้
// Verify Token Middleware
const verifyToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res
      .status(403)
      .json({ error: "No token provided or incorrect format" });
  }

  const token = authHeader.split(" ")[1];
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.userId = decoded.id; // Store the user ID for later use
    req.role = decoded.role; // Store the role for later use
    next(); // Proceed to the next middleware or route handler
  } catch (err) {
    return res.status(401).json({ error: "Unauthorized: Invalid token" });
  }
};

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = "uploads/";
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir);
    }
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    const uniqueName = crypto.randomBytes(16).toString("hex");
    const fileExtension = path.extname(file.originalname); // ดึงนามสกุลไฟล์ เช่น .jpg, .png
    const originalName = path.basename(file.originalname, fileExtension); // ดึงชื่อไฟล์ต้นฉบับ
    const timestamp = Date.now(); // เวลาปัจจุบันในหน่วย milliseconds

    // ตั้งชื่อไฟล์ใหม่ด้วย timestamp, original name, unique hash และ extension
    const newFileName = `${timestamp}_${originalName}_${uniqueName}${fileExtension}`;

    // แสดงชื่อไฟล์ใน console log เพื่อตรวจสอบ
    console.log(`File saved as: ${newFileName}`);

    cb(null, newFileName); // บันทึกชื่อไฟล์
  },
});

const upload = multer({
  storage: storage, // เปลี่ยนจาก dest เป็น storage ที่เราสร้างไว้
  limits: {
    fileSize: 2147483648,
  },
});

// Generate OTP
function generateOtp() {
  const otp = crypto.randomBytes(3).toString("hex"); // 3 bytes = 6 hex characters
  return parseInt(otp, 16).toString().slice(0, 4);
}

function sendOtpEmail(email, otp, callback) {
  const transporter = nodemailer.createTransport({
    service: "Gmail",
    auth: {
      user: process.env.email,
      pass: process.env.emailpassword,
    },
  });

  const mailOptions = {
    from: process.env.email,
    to: email,
    subject: "Your OTP Code",
    html: `
      <div style="font-family: Arial, sans-serif; color: #333;">
        <h2 style="color: #007bff;">Your OTP Code</h2>
        <p>Hello,</p>
        <p>We received a request to verify your email address. Please use the OTP code below to complete the process:</p>
        <div style="padding: 10px; border: 2px solid #007bff; display: inline-block; font-size: 24px; color: #007bff; font-weight: bold;">
          ${otp}
        </div>
        <p>This code will expire in 10 minutes.</p>
        <p>If you didnt request this, please ignore this email.</p>
        <p style="margin-top: 20px;">Thanks, <br> The Team</p>
        <hr>
        <p style="font-size: 12px; color: #999;">This is an automated email, please do not reply.</p>
      </div>
    `,
  };

  transporter.sendMail(mailOptions, (error, info) => {
    if (error) {
      console.error("Error sending OTP email:", error); // Log the error for debugging purposes
      return callback({
        error: "Failed to send OTP email. Please try again later.",
      });
    }
    callback(null, info); // Proceed if the email was successfully sent
  });
}

// --- Middleware for JWT verification ---
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (token == null) {
      console.log('Access Denied: No token provided.');
      return res.sendStatus(401); // No token
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
      if (err) {
          console.log('Forbidden: Invalid token.', err.message);
          return res.sendStatus(403); // Invalid token
      }
      req.user = user; // Attach user payload to request (contains id, role)
      next();
  });
};

// --- Middleware for Admin role check ---
const authorizeAdmin = (req, res, next) => {
  if (req.user && req.user.role === 'admin') {
      next();
  } else {
      console.log('Forbidden: Admin access required. User role:', req.user ? req.user.role : 'N/A');
      res.status(403).json({ message: "Forbidden: Admin access required" });
  }
};


//########################################################   Register API  #######################################################


// Register a new email user or reactivate if deactivated
app.post("/api/register/email", async (req, res) => {
  try {
    const { email } = req.body;

    // Check if the email is already registered and active
    const checkRegisteredSql =
      "SELECT * FROM users WHERE email = ? AND status = 'active' AND password IS NOT NULL";

    pool.query(checkRegisteredSql, [email], (err, results) => {
      if (err) throw new Error("Database error during email registration check");

      // If the email is already registered and active
      if (results.length > 0)
        return res.status(400).json({ error: "Email already registered" });

      // Check if the email exists but is deactivated
      const checkDeactivatedSql =
        "SELECT * FROM users WHERE email = ? AND status = 'deactivated'";

      pool.query(checkDeactivatedSql, [email], (err, deactivatedResults) => {
        if (err) throw new Error("Database error during email check");

        // If the email exists and is deactivated, reactivate the account
        if (deactivatedResults.length > 0) {
          const reactivateUserSql =
            "UPDATE users SET status = 'active' WHERE email = ?";
          pool.query(reactivateUserSql, [email], (err) => {
            if (err) throw new Error("Database error during account reactivation");

            return res.status(200).json({
              message: "Account reactivated successfully. You can now log in.",
            });
          });
        } else {
          // Check if the email is in use but the registration process was incomplete
          const checkIncompleteSql =
            "SELECT * FROM users WHERE email = ? AND password IS NULL AND status = 'active' ";
          pool.query(checkIncompleteSql, [email], (err, results) => {
            if (err) throw new Error("Database error during email check");
            if (results.length > 0)
              return res
                .status(400)
                .json({
                  error: "Email already in use or used in another sign-in",
                });

            // If no existing user found, proceed with OTP generation
            const otp = generateOtp();
            const expiresAt = new Date(Date.now() + 10 * 60 * 1000);

            const findOtpSql = "SELECT * FROM otps WHERE email = ?";
            pool.query(findOtpSql, [email], (err, otpResults) => {
              if (err) throw new Error("Database error during OTP retrieval");

              if (otpResults.length > 0) {
                const updateOtpSql =
                  "UPDATE otps SET otp = ?, expires_at = ? WHERE email = ?";
                pool.query(updateOtpSql, [otp, expiresAt, email], (err) => {
                  if (err) throw new Error("Database error during OTP update");
                  sendOtpEmail(email, otp, (error) => {
                    if (error) throw new Error("Error sending OTP email");
                    res.status(200).json({ message: "OTP sent to email" });
                  });
                });
              } else {
                const insertOtpSql =
                  "INSERT INTO otps (email, otp, expires_at) VALUES (?, ?, ?)";
                pool.query(insertOtpSql, [email, otp, expiresAt], (err) => {
                  if (err) throw new Error("Database error during OTP insertion");
                  sendOtpEmail(email, otp, (error) => {
                    if (error) throw new Error("Error sending OTP email");
                    res.status(200).json({ message: "OTP sent to email" });
                  });
                });
              }
            });
          });
        }
      });
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Verify OTP
app.post("/api/register/verify-otp", async (req, res) => {
  try {
    const { email, otp } = req.body;
    const verifyOtpSql =
      "SELECT otp, expires_at FROM otps WHERE email = ? AND otp = ?";

    pool.query(verifyOtpSql, [email, otp], (err, results) => {
      if (err) throw new Error("Database error during OTP verification");
      if (results.length === 0)
        return res.status(400).json({ error: "Invalid OTP" });

      const { expires_at } = results[0];
      const now = new Date();

      if (now > new Date(expires_at))
        return res.status(400).json({ error: "OTP has expired" });

      res
        .status(200)
        .json({ message: "OTP verified, you can set your password now" });
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Register User
app.post("/api/register/set-password", async (req, res) => {
  try {
    const { email, password } = req.body;
    const hash = await bcrypt.hash(password, 10);

    const sql = "INSERT INTO users (email, password, status, role, username, bio) VALUES (?, ?, 'active', 'user', '', 'My bio....')";
    pool.query(sql, [email, hash], (err) => {
      if (err) throw new Error("Database error during registration");

      const deleteOtpSql = "DELETE FROM otps WHERE email = ?";
      pool.query(deleteOtpSql, [email], (err) => {
        if (err) throw new Error("Database error during OTP cleanup");
        res.status(201).json({ message: "User registered successfully" });
      });
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Resend OTP for Registration
app.post("/api/resend-otp/register", async (req, res) => {
  try {
    const { email } = req.body;
    const findOtpSql = "SELECT otp, expires_at FROM otps WHERE email = ?";

    pool.query(findOtpSql, [email], (err, results) => {
      if (err) throw new Error("Database error during OTP lookup");
      if (results.length === 0)
        return res
          .status(400)
          .json({
            error: "No OTP found for this email. Please register first.",
          });

      const { otp, expires_at } = results[0];
      const now = new Date();

      if (now > new Date(expires_at)) {
        const newOtp = generateOtp();
        const newExpiresAt = new Date(now.getTime() + 10 * 60 * 1000);
        const updateOtpSql =
          "UPDATE otps SET otp = ?, expires_at = ? WHERE email = ?";
        pool.query(updateOtpSql, [newOtp, newExpiresAt, email], (err) => {
          if (err) throw new Error("Database error during OTP update");
          sendOtpEmail(email, newOtp, (error) => {
            if (error) throw new Error("Error sending OTP email");
            res.status(200).json({ message: "New OTP sent to email" });
          });
        });
      } else {
        sendOtpEmail(email, otp, (error) => {
          if (error) throw new Error("Error resending OTP email");
          res.status(200).json({ message: "OTP resent to email" });
        });
      }
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Forgot Password
app.post("/api/forgot-password", async (req, res) => {
  try {
    const { email } = req.body;
    const userCheckSql =
      "SELECT * FROM users WHERE email = ? AND password IS NOT NULL AND status = 'active'";

    pool.query(userCheckSql, [email], (err, userResults) => {
      if (err) throw new Error("Database error during email check");
      if (userResults.length === 0)
        return res.status(400).json({ error: "Email not found" });

      const otp = generateOtp();
      const expiresAt = new Date(Date.now() + 10 * 60 * 1000);

      const otpCheckSql = "SELECT * FROM password_resets WHERE email = ?";
      pool.query(otpCheckSql, [email], (err, otpResults) => {
        if (err) throw new Error("Database error during OTP check");

        if (otpResults.length > 0) {
          const updateOtpSql =
            "UPDATE password_resets SET otp = ?, expires_at = ? WHERE email = ?";
          pool.query(updateOtpSql, [otp, expiresAt, email], (err) => {
            if (err) throw new Error("Database error during OTP update");
            sendOtpEmail(email, otp, (error) => {
              if (error) throw new Error("Error sending OTP email");
              res.status(200).json({ message: "OTP sent to email" });
            });
          });
        } else {
          const saveOtpSql =
            "INSERT INTO password_resets (email, otp, expires_at) VALUES (?, ?, ?)";
          pool.query(saveOtpSql, [email, otp, expiresAt], (err) => {
            if (err) throw new Error("Database error during OTP save");
            sendOtpEmail(email, otp, (error) => {
              if (error) throw new Error("Error sending OTP email");
              res.status(200).json({ message: "OTP sent to email" });
            });
          });
        }
      });
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Verify Reset OTP
app.post("/api/verify-reset-otp", async (req, res) => {
  try {
    const { email, otp } = req.body;

    if (!email || !otp)
      return res.status(400).json({ error: "Email and OTP are required" });

    const verifyOtpSql =
      "SELECT otp, expires_at FROM password_resets WHERE email = ? AND otp = ?";
    pool.query(verifyOtpSql, [email, otp], (err, results) => {
      if (err) throw new Error("Database error during OTP verification");
      if (results.length === 0)
        return res.status(400).json({ error: "Invalid OTP or email" });

      const { expires_at } = results[0];
      const now = new Date();

      if (now > new Date(expires_at))
        return res.status(400).json({ error: "OTP has expired" });

      res
        .status(200)
        .json({ message: "OTP is valid, you can set a new password" });
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Reset Password
app.post("/api/reset-password", async (req, res) => {
  try {
    const { email, newPassword } = req.body;
    const hashedPassword = await bcrypt.hash(newPassword, 10);

    const updatePasswordSql = "UPDATE users SET password = ? WHERE email = ?";
    pool.query(updatePasswordSql, [hashedPassword, email], (err) => {
      if (err) throw new Error("Database error during password update");

      const deleteOtpSql = "DELETE FROM password_resets WHERE email = ?";
      pool.query(deleteOtpSql, [email], (err) => {
        if (err) throw new Error("Database error during OTP deletion");
        res
          .status(200)
          .json({ message: "Password has been updated successfully" });
      });
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Resend OTP for Reset Password
app.post("/api/resent-otp/reset-password", async (req, res) => {
  try {
    const { email } = req.body;
    const userCheckSql = "SELECT * FROM users WHERE email = ?";

    pool.query(userCheckSql, [email], (err, userResults) => {
      if (err) throw new Error("Database error during email check");
      if (userResults.length === 0)
        return res.status(400).json({ error: "Email not found" });

      const otpCheckSql = "SELECT * FROM password_resets WHERE email = ?";
      pool.query(otpCheckSql, [email], (err, otpResults) => {
        if (err) throw new Error("Database error during OTP check");
        if (otpResults.length === 0)
          return res
            .status(400)
            .json({ error: "No OTP record found for this email" });

        const otp = generateOtp();
        const expiresAt = new Date(Date.now() + 10 * 60 * 1000);

        const updateOtpSql =
          "UPDATE password_resets SET otp = ?, expires_at = ? WHERE email = ?";
        pool.query(updateOtpSql, [otp, expiresAt, email], (err) => {
          if (err) throw new Error("Database error during OTP update");
          sendOtpEmail(email, otp, (error) => {
            if (error) throw new Error("Error sending OTP email");
            res.status(200).json({ message: "New OTP sent to email" });
          });
        });
      });
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


//########################################################   Login API  #######################################################


// Login
app.post("/api/login", async (req, res) => {
  try {
      const { email, password, google_id } = req.body; // รับ google_id เพิ่มเข้ามาด้วย

      // Get the users IP address (optional)
      const ipAddress = req.headers["x-forwarded-for"] || req.connection.remoteAddress;

      // ตรวจสอบว่ามีอีเมลส่งมาหรือไม่
      if (!email) {
          return res.status(400).json({ message: "Email is required." });
      }

      const sql = "SELECT * FROM users WHERE email = ?";
      pool.query(sql, [email], async (err, results) => { // ใช้ async-await ที่นี่
          if (err) {
              console.error("Database error during login:", err);
              return res.status(500).json({ error: "Database error during login" });
          }

          let user = results.length > 0 ? results[0] : null;

          // --- กรณีที่ผู้ใช้กำลังพยายามล็อกอินด้วย Google ---
          if (google_id) {
              if (!user) {
                  // ถ้าไม่พบ user ด้วยอีเมลนี้เลย -> สร้าง user ใหม่ด้วย Google ID
                  const insertSql = "INSERT INTO users (email, google_id, created_at, last_login, last_login_ip, status, role, failed_attempts) VALUES (?, ?, NOW(), NOW(), ?, ?, ?, ?)";
                  pool.query(insertSql, [email, google_id, ipAddress, 'active', 'user', 0], (insertErr, insertResult) => {
                      if (insertErr) {
                          console.error("Error creating new user with Google ID:", insertErr);
                          return res.status(500).json({ message: "Failed to create user with Google ID." });
                      }
                      user = {
                          id: insertResult.insertId,
                          email: email,
                          username: null, // หรือค่า default อื่นๆ
                          picture: null,
                          google_id: google_id,
                          password: null,
                          status: 'active',
                          role: 'user',
                          failed_attempts: 0,
                          last_login: new Date(),
                          last_login_ip: ipAddress,
                      };
                      // ส่ง Token กลับทันที
                      const token = jwt.sign({ id: user.id, role: user.role }, JWT_SECRET);
                      return res.status(200).json({
                          message: "Google sign-in successful. New user created.",
                          token,
                          user: {
                              id: user.id,
                              email: user.email,
                              username: user.username,
                              picture: user.picture,
                              last_login: new Date(),
                              last_login_ip: ipAddress,
                          },
                      });
                  });
                  return; // ออกจากการทำงานของ API นี้
              }

              // ถ้าพบ user ด้วยอีเมลนี้
              if (user.google_id === null) {
                  // ถ้า user นี้ไม่มี google_id (เคยสมัครด้วย email/password มาก่อน) -> อัปเดต google_id ให้เขา
                  const updateGoogleIdSql = "UPDATE users SET google_id = ? WHERE id = ?";
                  pool.query(updateGoogleIdSql, [google_id, user.id], (updateErr) => {
                      if (updateErr) {
                          console.error("Error updating Google ID:", updateErr);
                          return res.status(500).json({ message: "Failed to link Google account." });
                      }
                      user.google_id = google_id; // อัปเดต object user ที่ใช้งานอยู่
                      // ดำเนินการล็อกอินต่อ
                      handleSuccessfulLogin(res, user, ipAddress);
                  });
              } else if (user.google_id === google_id) {
                  // ถ้า user มี google_id และตรงกัน -> ล็อกอินปกติ
                  handleSuccessfulLogin(res, user, ipAddress);
              } else {
                  // ถ้า user มี google_id แต่ไม่ตรงกัน (คนละ Google Account หรือซ้ำซ้อน)
                  return res.status(400).json({ message: "This email is already associated with a different Google account." });
              }
          }
          // --- จบกรณีล็อกอินด้วย Google ---

          // --- กรณีที่ผู้ใช้กำลังพยายามล็อกอินด้วย Email/Password ---
          else if (password) {
              if (!user) {
                  // ถ้าไม่พบ user ด้วยอีเมลนี้เลย (และไม่ได้มาจาก Google ID)
                  return res.status(404).json({ message: "No user found with this email." });
              }

              // ตรวจสอบว่า user มีรหัสผ่านหรือไม่
              if (user.password === null) {
                  // ถ้า user นี้ไม่มีรหัสผ่าน (เคยสมัครด้วย Google มาก่อน)
                  return res.status(400).json({ message: "Please sign in using Google or set a password for this account first." });
              }

              // Check if the user's status is active
              if (user.status !== 'active') {
                  return res.status(403).json({ message: "User is Suspended" });
              }

              // If the user has exceeded failed login attempts, block them for 5 minutes
              if (user.failed_attempts >= 5 && user.last_failed_attempt) {
                  const now = Date.now();
                  const timeSinceLastAttempt = now - new Date(user.last_failed_attempt).getTime();
                  if (timeSinceLastAttempt < 300000) { // 5 minutes
                      return res.status(429).json({
                          message: "Too many failed login attempts. Try again in 5 minutes.",
                      });
                  }
              }

              // Compare the entered password with the stored hashed password
              bcrypt.compare(password, user.password, (err, isMatch) => {
                  if (err) {
                      console.error("Password comparison error:", err);
                      return res.status(500).json({ error: "Password comparison error" });
                  }
                  if (!isMatch) {
                      // Increment failed attempts and update last_failed_attempt
                      const updateFailSql = "UPDATE users SET failed_attempts = failed_attempts + 1, last_failed_attempt = NOW() WHERE id = ?";
                      pool.query(updateFailSql, [user.id], (err) => {
                          if (err) console.error("Error logging failed login attempt:", err);
                      });
                      return res.status(401).json({ message: "Email or Password is incorrect." });
                  }

                  // Successful login
                  handleSuccessfulLogin(res, user, ipAddress);
              });
          }
          // --- จบกรณีล็อกอินด้วย Email/Password ---

          else {
              // ไม่มีทั้ง password และ google_id ส่งมา (request ไม่ถูกต้อง)
              return res.status(400).json({ message: "Missing login credentials (password or google_id)." });
          }
      });
  } catch (error) {
      console.error("Internal error:", error.message);
      res.status(500).json({ error: "Internal server error" });
  }
});


// Helper function สำหรับการล็อกอินสำเร็จ
function handleSuccessfulLogin(res, user, ipAddress) {
  const resetFailSql = "UPDATE users SET failed_attempts = 0, last_login = NOW(), last_login_ip = ? WHERE id = ?";
  pool.query(resetFailSql, [ipAddress, user.id], (err) => {
      if (err) {
          console.error("Error resetting failed attempts or updating login time:", err);
          return res.status(500).json({ error: "Error updating login status." });
      }

      // Generate JWT token
      const token = jwt.sign({ id: user.id, role: user.role }, JWT_SECRET);

      // Return successful login response with token and user data
      res.status(200).json({
          message: "Authentication successful",
          token,
          user: {
              id: user.id,
              email: user.email,
              username: user.username, // อาจจะเป็น null ถ้าเพิ่งสร้าง
              picture: user.picture,   // อาจจะเป็น null ถ้าเพิ่งสร้าง
              last_login: new Date(),
              last_login_ip: ipAddress,
          },
      });
  });
}


// Set profile route (Profile setup or update)
app.post("/api/set-profile", verifyToken, upload.single('picture'), (req, res) => {
  const { newUsername, birthday, gender } = req.body; // <<-- เพิ่ม gender เข้ามา
  const userId = req.userId;
  const picture = req.file ? `/uploads/${req.file.filename}` : null; 

  // <<-- เพิ่ม gender เข้าไปในเงื่อนไขการตรวจสอบข้อมูลที่จำเป็น
  if (!newUsername || !picture || !birthday || !gender) {
    return res.status(400).json({ message: "New username, picture, birthday, and gender are required" });
  }

  // Convert birthday from DD/MM/YYYY to YYYY-MM-DD
  const birthdayParts = birthday.split('/');
  // ตรวจสอบความถูกต้องของรูปแบบวันที่ก่อนแปลง
  if (birthdayParts.length !== 3 || isNaN(parseInt(birthdayParts[0])) || isNaN(parseInt(birthdayParts[1])) || isNaN(parseInt(birthdayParts[2]))) {
    return res.status(400).json({ message: "Invalid birthday format. Please use DD/MM/YYYY" });
  }
  const formattedBirthday = `${birthdayParts[2]}-${birthdayParts[1]}-${birthdayParts[0]}`;

  // <<-- คำนวณอายุ
  let age = null;
  try {
    const birthDate = new Date(formattedBirthday);
    const today = new Date();
    age = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    if (age < 0) { // กรณีวันเกิดในอนาคต หรือผิดพลาด
        age = 0; // ตั้งค่าเป็น 0 หรือส่ง error ก็ได้
    }
  } catch (e) {
    console.error("Error calculating age:", e);
    // ไม่ได้ return error ทันที เพราะอาจจะยังต้องการอัปเดตข้อมูลอื่น
    // หรือจะ return res.status(400).json({ message: "Invalid birthday date" }); ก็ได้ถ้าต้องการบังคับ
  }

  // <<-- ตรวจสอบค่า gender ให้ถูกต้อง
  const allowedGenders = ['Male', 'Female', 'Other'];
  if (!allowedGenders.includes(gender)) {
    return res.status(400).json({ message: "Invalid gender value. Must be Male, Female, or Other." });
  }

  // Check if the new username is already taken
  const checkUsernameQuery = "SELECT * FROM users WHERE username = ?";
  pool.query(checkUsernameQuery, [newUsername], (err, results) => {
    if (err) {
      console.error("Error checking username: ", err);
      return res.status(500).json({ message: "Database error checking username" });
    }

    if (results.length > 0) {
      return res.status(400).json({ message: "Username already taken" });
    }

    // Update the profile with the new username, picture (with '/uploads/'), birthday (formatted), gender, and age
    const updateProfileQuery = "UPDATE users SET username = ?, picture = ?, birthday = ?, gender = ?, age = ? WHERE id = ?"; // <<-- เพิ่ม gender และ age ใน query
    pool.query(updateProfileQuery, [newUsername, picture, formattedBirthday, gender, age, userId], (err) => { // <<-- เพิ่ม gender และ age ใน parameters
      if (err) {
        console.error("Error updating profile: ", err);
        return res.status(500).json({ message: "Error updating profile" });
      }

      return res.status(200).json({ message: "Profile set/updated successfully" });
    });
  });
});


// Google Sign-In with soft delete handling
app.post("/api/google-signin", async (req, res) => {
  try {
    const { googleId, email } = req.body;

    // ตรวจสอบว่ามีการส่ง Google ID และ Email เข้ามาหรือไม่
    if (!googleId || !email) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    // ค้นหาผู้ใช้ที่มี google_id และ status = 'active' หรือ 'deleted'
    const checkGoogleIdSql =
      "SELECT * FROM users WHERE google_id = ? AND (status = 'active' OR status = 'deactivated')";
    pool.query(checkGoogleIdSql, [googleId], (err, googleIdResults) => {
      if (err) {
        console.error("Original database error during Google ID check:", err); // เพิ่มบรรทัดนี้
        throw new Error("Database error during Google ID check");
      }

      if (googleIdResults.length > 0) {
        const user = googleIdResults[0];

        // Reactivate user if status is 'deleted'
        if (user.status === "deactivated") {
          const reactivateSql = "UPDATE users SET status = 'active', email = ? WHERE google_id = ?";
          pool.query(reactivateSql, [email, googleId], (err) => {
            if (err) {
              console.error("Original database error during user reactivation:", err); // เพิ่มบรรทัดนี้
              throw new Error("Database error during user reactivation");
            }

            const token = jwt.sign({ id: user.id, role: user.role }, JWT_SECRET);
            return res.json({
              message: "User reactivated and authenticated successfully",
              token,
              user: {
                id: user.id,
                email: user.email,
                picture: user.picture,
                username: user.username,
                google_id: user.google_id,
                role: user.role,
                status: 'active',
              },
            });
          });
        } else {
          // If the user is already active, update email if necessary
          const updateSql = "UPDATE users SET email = ? WHERE google_id = ?";
          pool.query(updateSql, [email, googleId], (err) => {
            if (err) {
              console.error("Original database error during user update:", err); // เพิ่มบรรทัดนี้
              throw new Error("Database error during user update");
            }

            const token = jwt.sign({ id: user.id, role: user.role }, JWT_SECRET);
            return res.json({
              message: "User information updated successfully",
              token,
              user: {
                id: user.id,
                email: user.email,
                picture: user.picture,
                username: user.username,
                google_id: user.google_id,
                role: user.role,
                status: user.status,
              },
            });
          });
        }
      } else {
        // ตรวจสอบว่ามี email นี้ในฐานข้อมูลหรือไม่ (และเป็น active)
        const checkEmailSql = "SELECT * FROM users WHERE email = ? AND status = 'active'";
        pool.query(checkEmailSql, [email], (err, emailResults) => {
          if (err) {
            console.error("Original database error during email check:", err); // เพิ่มบรรทัดนี้
            throw new Error("Database error during email check");
          }
          if (emailResults.length > 0) {
            // ถ้า email นี้ถูกใช้งานอยู่แล้วโดยบัญชีอื่น (ที่ไม่ใช่ Google ID นี้)
            return res.status(409).json({
              error: "Email already registered with another account",
            });
          }

          // หากไม่มีผู้ใช้ในระบบ ให้สร้างผู้ใช้ใหม่ด้วย Google ID, email, status และ role
          const insertSql =
            "INSERT INTO users (google_id, email, username, status, role) VALUES (?, ?, '', 'active', 'user')";
          pool.query(insertSql, [googleId, email], (err, result) => {
            // นี่คือบรรทัดที่ 756 ที่งับเจอ Error:
            if (err) {
              console.error("Original database error during user insertion:", err); // <<-- เพิ่มบรรทัดนี้
              throw new Error("Database error during user insertion");
            }

            const newUserId = result.insertId;
            const newUserSql = "SELECT * FROM users WHERE id = ?";
            pool.query(newUserSql, [newUserId], (err, newUserResults) => {
              if (err) {
                console.error("Original database error during new user fetch:", err); // เพิ่มบรรทัดนี้
                throw new Error("Database error during new user fetch");
              }

              const newUser = newUserResults[0];
              const token = jwt.sign({ id: newUser.id, role: newUser.role }, JWT_SECRET);

              return res.status(201).json({
                message: "User registered and authenticated successfully",
                token,
                user: {
                  id: newUser.id,
                  email: newUser.email,
                  picture: newUser.picture,
                  username: newUser.username,
                  google_id: newUser.google_id,
                  role: newUser.role,
                  status: newUser.status,
                },
              });
            });
          });
        });
      }
    });
  } catch (error) {
    console.error("Caught error in Google Sign-In API:", error.message); // แก้ไขข้อความตรงนี้ด้วยก็ได้
    res.status(500).json({ error: "Internal server error" });
  }
});


//########################################################   Interactions API  #######################################################


// POST /api/interactions - บันทึกการโต้ตอบใหม่
app.post("/api/interactions", verifyToken, async (req, res) => {
    const { post_id, action_type, content } = req.body;
    const user_id = req.userId; // ดึง userId จาก Token
  
    // ตรวจสอบข้อมูลที่ส่งมาว่าไม่ว่างเปล่า
    const postIdValue = post_id ? post_id : null;
  
    if (!user_id || !action_type) {
      return res
        .status(400)
        .json({ error: "Missing required fields: user_id or action_type" });
    }
  
    // ส่วนนี้ทำงานได้อยู่แล้ว
    const insertSql = `
    INSERT INTO user_interactions (user_id, post_id, action_type, content)
    VALUES (?, ?, ?, ?);
  `;
    const values = [user_id, postIdValue, action_type, content || null];
  
    pool.query(insertSql, values, (error, results) => {
      if (error) {
        console.error("Database error:", error);
        return res.status(500).json({ error: "Error saving interaction" });
      }
      res
        .status(201)
        .json({
          message: "Interaction saved successfully",
          interaction_id: results.insertId,
        });
    });
  });


// GET /api/interactions - ดึงข้อมูลการโต้ตอบทั้งหมด
app.get("/api/interactions", verifyToken, async (req, res) => {
  const fetchSql = `
    SELECT 
        ui.id, 
        u.username, 
        p.content AS post_content, 
        ui.action_type, 
        ui.content AS interaction_content, 
        ui.created_at 
    FROM user_interactions ui
    JOIN users u ON ui.user_id = u.id
    JOIN posts p ON ui.post_id = p.id
    ORDER BY ui.created_at DESC;
  `;

  pool.query(fetchSql, (error, results) => {
    if (error) {
      console.error("Database error:", error);
      return res.status(500).json({ error: "Error fetching interactions" });
    }
    res.json(results);
  });
});


// GET /api/interactions/user/:userId - ดึงข้อมูลการโต้ตอบของผู้ใช้แต่ละคน
app.get("/api/interactions/user/:userId", verifyToken, async (req, res) => {
  const { userId } = req.params;

  if (parseInt(req.userId) !== parseInt(userId)) {
    return res
      .status(403)
      .json({ error: "Unauthorized access: User ID does not match" });
  }

  const fetchUserInteractionsSql = `
    SELECT 
        ui.id, 
        u.username, 
        p.content AS post_content, 
        ui.action_type, 
        ui.content AS interaction_content, 
        ui.created_at 
    FROM user_interactions ui
    JOIN users u ON ui.user_id = u.id
    JOIN posts p ON ui.post_id = p.id
    WHERE ui.user_id = ?
    ORDER BY ui.created_at DESC;
  `;

  pool.query(fetchUserInteractionsSql, [userId], (error, results) => {
    if (error) {
      console.error("Database error:", error);
      return res
        .status(500)
        .json({ error: "Error fetching user interactions" });
    }
    res.json(results);
  });
});


// DELETE /api/interactions/:id - ลบข้อมูลการโต้ตอบตาม ID
app.delete("/api/interactions/:id", verifyToken, async (req, res) => {
  const { id } = req.params;

  const deleteSql =
    "DELETE FROM user_interactions WHERE id = ? AND user_id = ?";
  pool.query(deleteSql, [id, req.userId], (error, results) => {
    if (error) {
      console.error("Database error:", error);
      return res.status(500).json({ error: "Error deleting interaction" });
    }

    if (results.affectedRows === 0) {
      return res
        .status(404)
        .json({
          message:
            "Interaction not found or you are not authorized to delete this interaction",
        });
    }

    res.json({ message: "Interaction deleted successfully" });
  });
});


// PUT /api/interactions/:id - อัปเดตข้อมูลการโต้ตอบตาม ID
app.put("/api/interactions/:id", verifyToken, async (req, res) => {
  const { id } = req.params;
  const { action_type, content } = req.body;

  const updateSql = `
    UPDATE user_interactions 
    SET action_type = ?, content = ?, updated_at = NOW() 
    WHERE id = ? AND user_id = ?;
  `;
  const values = [action_type, content || null, id, req.userId];

  pool.query(updateSql, values, (error, results) => {
    if (error) {
      console.error("Database error:", error);
      return res.status(500).json({ error: "Error updating interaction" });
    }

    if (results.affectedRows === 0) {
      return res
        .status(404)
        .json({
          message:
            "Interaction not found or you are not authorized to update this interaction",
        });
    }

    res.json({ message: "Interaction updated successfully" });
  });
});


function isValidJson(str) {
  try {
    JSON.parse(str);
    return true;
  } catch (e) {
    return false;
  }
}


// API สำหรับตรวจสอบสถานะการกดไลค์ของผู้ใช้
app.get("/api/checkLikeStatus/:postId/:userId", verifyToken, (req, res) => {
  const { postId, userId } = req.params;
  const user_id = req.userId;

  // ตรวจสอบสิทธิ์ว่าผู้ใช้มีสิทธิ์ในการเข้าถึงหรือไม่
  if (user_id != userId) {
    return res
      .status(403)
      .json({ error: "Unauthorized access: User ID does not match" });
  }

  // ตรวจสอบว่า postId และ userId มีค่า
  if (!postId || !userId) {
    return res
      .status(400)
      .json({ error: "Missing required parameters: postId or userId" });
  }

  // SQL Query เพื่อเช็คสถานะการกดไลค์ในตาราง likes
  const query = `
    SELECT COUNT(*) AS isLiked
    FROM likes 
    WHERE post_id = ? AND user_id = ?
  `;

  pool.query(query, [postId, userId], (err, results) => {
    if (err) {
      console.error("Database error during like status check:", err);
      return res
        .status(500)
        .json({ error: "Internal server error during like status check" });
    }

    // ตรวจสอบสถานะการกดไลค์ (ถ้าผลลัพธ์มากกว่า 0 แสดงว่ามีการกดไลค์)
    const isLiked = results[0].isLiked > 0;
    res.json({ isLiked });
  });
});


//########################################################   Post API  #######################################################


// View All Posts with Token Verification
app.get("/api/posts", verifyToken, (req, res) => {
  try {
    const userId = req.userId; // ดึง user_id จาก token ที่ผ่านการตรวจสอบแล้ว

    const query = `
      SELECT posts.*, users.username, users.picture, 
      (SELECT COUNT(*) FROM likes WHERE post_id = posts.id AND user_id = ?) AS is_liked
      FROM posts 
      JOIN users ON posts.user_id = users.id
      WHERE posts.status = 'active' 
      ORDER BY posts.updated_at DESC;
    `;

    pool.query(query, [userId], (err, results) => {
      if (err) {
        console.error("Database error during posts retrieval:", err);
        return res
          .status(500)
          .json({ error: "Internal server error during posts retrieval" });
      }

      const parsedResults = results.map((post) => {
        const photoUrls = Array.isArray(post.photo_url)
          ? post.photo_url.map((photo) => photo)
          : [];
        const videoUrls = Array.isArray(post.video_url)
          ? post.video_url.map((video) => video)
          : [];

        return {
          id: post.id,
          userId: post.user_id,
          title: post.Title,
          content: post.content,
          time: post.time,
          updated: post.updated_at,
          photo_url: photoUrls,
          video_url: videoUrls,
          userName: post.username,
          userProfileUrl: post.picture ? post.picture : null,
        };
      });

      res.json(parsedResults);
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


//update status posts
app.put("/api/posts/:id/status", verifyToken, (req, res) => {
  const postId = req.params.id;
  const roles = req.role;

  // ตรวจสอบบทบาท (role) และหยุดการทำงานถ้าบทบาทไม่ถูกต้อง
  if (roles !== "admin") {
    return res.status(403).json({ error: "You do not have permission to update status." });
  }

  // รับค่า status ที่จะอัปเดตมาจาก Body
  const { status } = req.body;

  const query = "UPDATE posts SET status = ? WHERE id = ?";
  pool.query(query, [status, postId], (err, results) => {
    if (err) {
      console.error("Database error during post status update:", err);
      return res.status(500).json({ error: "Internal server error" });
    }

    // ตรวจสอบว่ามีการอัปเดตข้อมูลหรือไม่
    if (results.affectedRows === 0) {
      return res.status(404).json({ error: "Post not found or status not changed." });
    }

    res.json({ message: "Post status updated successfully." });
  });
});


// View a Single Post with Like and Comment Count and Show Comments
app.get("/api/posts/:id", verifyToken, (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.userId; // ดึง user_id จาก token ที่ผ่านการตรวจสอบแล้ว

    const queryPost = `
      SELECT p.*, u.username, u.picture, 
      (SELECT COUNT(*) FROM likes WHERE post_id = ?) AS like_count,
      (SELECT COUNT(*) FROM comments WHERE post_id = ?) AS comment_count,
      (SELECT COUNT(*) FROM likes WHERE post_id = ? AND user_id = ?) AS is_liked
      FROM posts p
      JOIN users u ON p.user_id = u.id 
      WHERE p.id = ?;
    `;

    const queryComments = `
      SELECT c.*, u.username, u.picture AS user_profile
      FROM comments c
      JOIN users u ON c.user_id = u.id
      WHERE c.post_id = ?;
    `;

    pool.query(queryPost, [id, id, id, userId, id], (err, postResults) => {
      if (err) {
        console.error("Database error during post retrieval:", err);
        return res
          .status(500)
          .json({ error: "Internal server error during post retrieval" });
      }

      if (postResults.length === 0) {
        return res.status(404).json({ error: "Post not found" });
      }

      const post = postResults[0];
      console.log("Post data fetched:", post); // เพิ่ม log เพื่อตรวจสอบข้อมูลโพสต์

      post.photo_url = isValidJson(post.photo_url)
        ? JSON.parse(post.photo_url)
        : [post.photo_url];
      post.video_url = isValidJson(post.video_url)
        ? JSON.parse(post.video_url)
        : [post.video_url];
      post.is_liked = post.is_liked > 0; // แปลงค่า is_liked ให้เป็น boolean

      pool.query(queryComments, [id], (err, commentResults) => {
        if (err) {
          console.error("Database error during comments retrieval:", err);
          return res
            .status(500)
            .json({ error: "Internal server error during comments retrieval" });
        }

        console.log("Comment data fetched:", commentResults); // เพิ่ม log เพื่อตรวจสอบข้อมูลคอมเมนต์

        res.json({
          ...post,
          like_count: post.like_count,
          productName: post.ProductName,
          comment_count: post.comment_count,
          update: post.updated_at,
          is_liked: post.is_liked, // เพิ่มสถานะการไลค์ของผู้ใช้ในข้อมูลโพสต์
          comments: commentResults.map((comment) => ({
            id: comment.id,
            user_id: comment.user_id,
            content: comment.comment_text,
            created_at: comment.created_at,
            username: comment.username,
            user_profile: comment.user_profile ? comment.user_profile : null,
          })),
        });
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Delete a Post
app.delete("/api/posts/:id", verifyToken, (req, res) => {
  const { id } = req.params;
  const user_id = req.userId; // Get user ID from the token

  // Check if the post belongs to the user
  const postCheckSql = "SELECT * FROM posts WHERE id = ? AND user_id = ?";
  pool.query(postCheckSql, [id, user_id], (postError, postResults) => {
      if (postError) {
          console.error("Database error during post check:", postError);
          return res.status(500).json({ error: "Database error during post check" });
      }
      if (postResults.length === 0) {
          return res.status(404).json({ error: "Post not found or you are not the owner" });
      }

      // Delete notifications related to the post
      const deleteNotificationsSql = "DELETE FROM notifications WHERE post_id = ?";
      pool.query(deleteNotificationsSql, [id], (deleteNotificationError) => {
          if (deleteNotificationError) {
              console.error("Database error during notification deletion:", deleteNotificationError);
              return res.status(500).json({ error: "Database error during notification deletion" });
          }

          // Delete the post
          const deletePostSql = "DELETE FROM posts WHERE id = ? AND user_id = ?";
          pool.query(deletePostSql, [id, user_id], (deletePostError, deletePostResults) => {
              if (deletePostError) {
                  console.error("Database error during post deletion:", deletePostError);
                  return res.status(500).json({ error: "Database error during post deletion" });
              }

              if (deletePostResults.affectedRows === 0) {
                  return res.status(404).json({ error: "Post not found or you are not the owner" });
              }

              res.json({ message: "Post deleted successfully" });
          });
      });
  });
});


app.get("/api/type", verifyToken, (req, res) => {
  const sqlQuery = "SELECT * FROM category";

  // Get a connection from the pool and query the database
  pool.getConnection((err, connection) => {
    if (err) {
      return res.status(500).json({ error: "Error connecting to the database" });
    }

    // Execute the query
    connection.query(sqlQuery, (err, result) => {
      // Release the connection back to the pool after query execution
      connection.release();

      if (err) {
        return res.status(500).json({ error: "Database query failed" });
      }

      // Send back the results
      res.json(result);
    });
  });
});


// API สำหรับกด like หรือ unlike โพสต์
app.post("/api/posts/like/:id", verifyToken, (req, res) => {
  const { id } = req.params; // Post ID จาก URL
  const { user_id } = req.body; // User ID จาก body ของ request

  try {
    // ตรวจสอบว่า userId ใน token ตรงกับ user_id ใน body หรือไม่
    if (parseInt(req.userId) !== parseInt(user_id)) {
      return res
        .status(403)
        .json({ error: "You are not authorized to like this post" });
    }

    // ตรวจสอบว่าโพสต์นั้นมีอยู่ในฐานข้อมูลหรือไม่
    const checkPostSql = "SELECT * FROM posts WHERE id = ?";
    pool.query(checkPostSql, [id], (err, postResults) => {
      if (err) {
        console.error("Database error during post check:", err);
        return res
          .status(500)
          .json({ error: "Database error during post check" });
      }
      if (postResults.length === 0) {
        return res.status(404).json({ error: "Post not found" });
      }

      // ตรวจสอบว่า user ได้กด like โพสต์นี้แล้วหรือยัง
      const checkLikeSql =
        "SELECT * FROM likes WHERE post_id = ? AND user_id = ?";
      pool.query(checkLikeSql, [id, user_id], (err, likeResults) => {
        if (err) {
          console.error("Database error during like check:", err);
          return res
            .status(500)
            .json({ error: "Database error during like check" });
        }

        if (likeResults.length > 0) {
          // ถ้าผู้ใช้กด like แล้ว ให้ unlike (ลบ like ออก)
          const unlikeSql =
            "DELETE FROM likes WHERE post_id = ? AND user_id = ?";
          pool.query(unlikeSql, [id, user_id], (err) => {
            if (err) {
              console.error("Database error during unlike:", err);
              return res
                .status(500)
                .json({ error: "Database error during unlike" });
            }

            // หลังจาก unlike เสร็จ ให้ดึงค่า likeCount ใหม่
            const likeCountQuery =
              "SELECT COUNT(*) AS likeCount FROM likes WHERE post_id = ?";
            pool.query(likeCountQuery, [id], (err, countResults) => {
              if (err) {
                console.error("Database error during like count:", err);
                return res
                  .status(500)
                  .json({ error: "Database error during like count" });
              }
              const likeCount = countResults[0].likeCount;
              res
                .status(200)
                .json({
                  message: "Post unliked successfully",
                  status: "unliked",
                  liked: false,
                  likeCount,
                });
            });
          });
        } else {
          // ถ้ายังไม่กด like ให้เพิ่มการ like
          const likeSql = "INSERT INTO likes (post_id, user_id) VALUES (?, ?)";
          pool.query(likeSql, [id, user_id], (err) => {
            if (err) {
              console.error("Database error during like:", err);
              return res
                .status(500)
                .json({ error: "Database error during like" });
            }

            // หลังจาก like เสร็จ ให้ดึงค่า likeCount ใหม่
            const likeCountQuery =
              "SELECT COUNT(*) AS likeCount FROM likes WHERE post_id = ?";
            pool.query(likeCountQuery, [id], (err, countResults) => {
              if (err) {
                console.error("Database error during like count:", err);
                return res
                  .status(500)
                  .json({ error: "Database error during like count" });
              }
              const likeCount = countResults[0].likeCount;
              res
                .status(201)
                .json({
                  message: "Post liked successfully",
                  status: "liked",
                  liked: true,
                  likeCount,
                });
            });
          });
        }
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Serve static files (uploaded images and videos)
app.use("/uploads", express.static(path.join(__dirname, "uploads")));


// Search API with grouped results by username, and include only the first photo_url
app.get("/api/search", (req, res) => {
  const { query } = req.query;

//แก้5

  if (!query) {
    return res.status(400).json({ error: "Search query is required" });
  }

  // Trim the query and convert to lowercase
  const searchValue = `%${query.trim().toLowerCase()}%`;

  // SQL query to search users and posts
  const searchSql = `
SELECT 
      u.id AS user_id,                
      u.username, 
      u.picture,
      p.id AS post_id,                
      LEFT(p.content, 100) AS content_preview,  
      p.title,                        
      p.photo_url                     
FROM users u
LEFT JOIN posts p ON p.user_id = u.id
WHERE (LOWER(u.username) LIKE ? 
       OR LOWER(p.content) LIKE ? 
       OR LOWER(p.title) LIKE ?)
  AND u.status = 'active'        
ORDER BY p.updated_at DESC;

  `;

  pool.query(
    searchSql,
    [searchValue, searchValue, searchValue],
    (err, results) => {
      if (err) {
        console.error("Database error during search:", err);
        return res.status(500).json({ error: "Internal server error" });
      }

      if (results.length === 0) {
        return res.status(404).json({ message: "No results found" });
      }

      // Group the results by username and aggregate their posts
      const groupedResults = results.reduce((acc, post) => {
        const username = post.username;

        // ตรวจสอบว่ามีโพสต์หรือไม่
        const hasPost = post.post_id !== null;

        // Check if the username already exists in the accumulator (grouped results)
        const existingUser = acc.find((user) => user.username === username);

        if (existingUser) {
          // ถ้ามีโพสต์ ให้เพิ่มข้อมูลโพสต์
          if (hasPost) {
            existingUser.posts.push({
              post_id: post.post_id,
              title: post.title,
              content_preview: post.content_preview,
              photo_url: post.photo_url || "",
            });
          }
        } else {
          // ถ้าไม่มีโพสต์ แสดงเฉพาะข้อมูลผู้ใช้
          acc.push({
            user_id: post.user_id,
            username: post.username,
            profile_image: post.picture,
            posts: hasPost
              ? [
                  {
                    post_id: post.post_id,
                    title: post.title,
                    content_preview: post.content_preview,
                    photo_url: post.photo_url || "",
                  },
                ]
              : undefined, // ไม่ต้องมี posts ถ้าไม่มีโพสต์
          });
        }

        return acc;
      }, []);

      // ส่งข้อมูล groupedResults กลับในรูปแบบ JSON
      res.json({
        results: groupedResults.map((user) => {
          // ลบ posts ถ้าไม่มีโพสต์
          if (!user.posts) {
            delete user.posts;
          }
          return user;
        }),
      });
    }
  );
});


app.get("/api/users/:userId/profile", verifyToken, (req, res) => {
  const userId = req.params.userId;

  if (req.userId.toString() !== userId) {
    return res
      .status(403)
      .json({ error: "You are not authorized to view this profile" });
  }

  // SQL query to get user profile and count posts
  const sql = `
      SELECT 
      u.id AS userId, 
      u.username, 
      u.picture AS profileImageUrl,
      u.bio,
      u.email,
      u.birthday,
      u.gender, 
      COUNT(p.id) AS post_count,
      (SELECT COUNT(*) FROM follower_following WHERE following_id = u.id) AS follower_count, 
      (SELECT COUNT(*) FROM follower_following WHERE follower_id = u.id) AS following_count   
      FROM users u
      LEFT JOIN posts p ON p.user_id = u.id
      WHERE u.id = ?
      GROUP BY u.id;
  `;

  // SQL query to get user posts
  const postSql = `
    SELECT 
      p.id AS post_id, 
      p.content, 
      p.photo_url, 
      p.video_url, 
      p.updated_at,
      (SELECT COUNT(*) FROM likes WHERE post_id = p.id) AS like_count,
      (SELECT COUNT(*) FROM comments WHERE post_id = p.id) AS comment_count
    FROM posts p
    WHERE p.user_id = ?
    AND p.status = 'active'
    ORDER BY p.updated_at DESC;
  `;

  // Execute the first query to get user profile
  pool.query(sql, [userId], (error, results) => {
    if (error) {
      return res
        .status(500)
        .json({ error: "Database error while fetching user profile" });
    }
    if (results.length === 0) {
      return res.status(404).json({ error: "User not found" });
    }

    const userProfile = results[0];

    // Execute the second query to get user's posts
    pool.query(postSql, [userId], (postError, postResults) => {
      if (postError) {
        return res
          .status(500)
          .json({ error: "Database error while fetching user posts" });
      }

      // Construct the response
      const response = {
        userId: userProfile.userId,
        email: userProfile.email,
        username: userProfile.username,
        birthday: userProfile.birthday,
        profileImageUrl: userProfile.profileImageUrl,
        followerCount: userProfile.follower_count,
        followingCount: userProfile.following_count,
        postCount: userProfile.post_count,
        gender: userProfile.gender,
        bio: userProfile.bio,
        posts: postResults.map(post => ({
          post_id: post.post_id,
          content: post.content,
          photoUrl: post.photo_url,
          videoUrl: post.video_url,
          updatedAt: post.updated_at,
          likeCount: post.like_count,
          commentCount: post.comment_count
        }))
      };

      // Send the response
      res.json(response);
    });
  });
});


// ดูโปรไฟล์
app.get("/api/users/:userId/view-profile", verifyToken, (req, res) => {
  const { userId } = req.params;

  const profileSql = `
    SELECT 
      u.id AS userId, 
      u.username, 
      u.picture AS profileImageUrl,
      u.bio,
      u.gender, 
      COUNT(p.id) AS post_count,
      (SELECT COUNT(*) FROM follower_following WHERE following_id = u.id) AS follower_count, 
      (SELECT COUNT(*) FROM follower_following WHERE follower_id = u.id) AS following_count   
    FROM users u
    LEFT JOIN posts p ON p.user_id = u.id
    WHERE u.id = ?
    GROUP BY u.id;
  `;

//แก้ 2 

  const postSql = `
    SELECT 
      p.id AS post_id,
      p.title, 
      p.content, 
      p.photo_url, 
      p.video_url, 
      p.updated_at,
      (SELECT COUNT(*) FROM likes WHERE post_id = p.id) AS like_count,
      (SELECT COUNT(*) FROM comments WHERE post_id = p.id) AS comment_count
    FROM posts p
    WHERE p.user_id = ?
    AND p.status = 'active'
    ORDER BY p.updated_at DESC;
  `;

  pool.query(profileSql, [userId], (error, profileResults) => {
    if (error) {
      return res
        .status(500)
        .json({ error: "Database error while fetching user profile" });
    }
    if (profileResults.length === 0) {
      return res.status(404).json({ error: "User not found" });
    }

    const userProfile = profileResults[0];

    pool.query(postSql, [userId], (postError, postResults) => {
      if (postError) {
        console.error("Database error while fetching user posts:", postError);
        return res
          .status(500)
          .json({ error: "Database error while fetching user posts" });
      }

      // ตรวจสอบและแปลง photo_url และ video_url ให้เป็น JSON Array
      const formattedPosts = postResults.map((post) => {
        let photos = [];
        let videos = [];

        // ตรวจสอบว่า `photo_url` เป็นอาร์เรย์อยู่แล้วหรือไม่
        if (Array.isArray(post.photo_url)) {
          photos = post.photo_url; // หากเป็นอาร์เรย์ ให้ใช้ข้อมูลตรง ๆ
        } else if (typeof post.photo_url === "string") {
          try {
            photos = JSON.parse(post.photo_url); // กรณีที่เป็นสตริง JSON Array ให้แปลงเป็นอาร์เรย์
          } catch (e) {
            console.error("Error parsing photo_url:", e.message);
          }
        }

        // ตรวจสอบว่า `video_url` เป็นอาร์เรย์อยู่แล้วหรือไม่
        if (Array.isArray(post.video_url)) {
          videos = post.video_url; // หากเป็นอาร์เรย์ ให้ใช้ข้อมูลตรง ๆ
        } else if (typeof post.video_url === "string") {
          try {
            videos = JSON.parse(post.video_url); // กรณีที่เป็นสตริง JSON Array ให้แปลงเป็นอาร์เรย์
          } catch (e) {
            console.error("Error parsing video_url:", e.message);
          }
        }

        return {
          post_id: post.post_id,
          title: post.title,
          content: post.content,
          created_at: post.updated_at,
          like_count: post.like_count,
          comment_count: post.comment_count,
          photos, // ส่งกลับ photos ที่ถูกแปลงเป็น Array แล้ว
          videos, // ส่งกลับ videos ที่ถูกแปลงเป็น Array แล้ว
        };
      });

      res.json({
        userId: userProfile.userId,
        username: userProfile.username,
        profileImageUrl: userProfile.profileImageUrl,
        followerCount: userProfile.follower_count,
        followingCount: userProfile.following_count,
        postCount: userProfile.post_count,
        gender: userProfile.gender,
        bio: userProfile.bio,
        posts: formattedPosts,
      });
    });
  });
});


//update profile
app.put("/api/users/:userId/profile",verifyToken,upload.single("profileImage"),(req, res) => {
    const userId = req.params.userId;

    // Extract the data from the request body
    let { username, bio, gender, birthday } = req.body;
    const profileImage = req.file ? `/uploads/${req.file.filename}` : null;

    // Validate that the necessary fields are provided
    if (!username || !bio || !gender || !birthday) {
      return res
        .status(400)
        .json({ error: "All fields are required: username, bio, gender, and birthday" });
    }

    // Check if birthday is valid and in the correct format
    if (isNaN(Date.parse(birthday))) {
      return res.status(400).json({ error: "Invalid birthday format" });
    }

    // Convert the birthday to the format "yyyy-MM-dd"
    birthday = formatDateForSQL(birthday); // This function should correctly format the date

    // Check if the username is already in use by another user
    const checkUsernameSql = `SELECT id FROM users WHERE username = ? AND id != ?`;

    pool.query(checkUsernameSql, [username, userId], (checkError, checkResults) => {
      if (checkError) {
        console.error("Error checking username:", checkError); // Logging the error
        return res.status(500).json({ error: "Database error while checking username" });
      }

      if (checkResults.length > 0) {
        // Username is already taken by another user
        return res.status(400).json({ error: "Username is already in use" });
      }

      // SQL query to update the user's profile
      let updateProfileSql = `UPDATE users SET username = ?, bio = ?, gender = ?, birthday = ?`;
      const updateData = [username, bio, gender, birthday];

      // If an image was uploaded, include the profileImage in the update
      if (profileImage) {
        updateProfileSql += `, picture = ?`;
        updateData.push(profileImage); // Insert profileImage into the query parameters
      }

      updateProfileSql += ` WHERE id = ?;`;
      updateData.push(userId); // Add the userId at the end of the updateData array

      // Execute the SQL query to update the user's profile
      pool.query(updateProfileSql, updateData, (error, results) => {
        if (error) {
          console.error("Error updating profile:", error); // Logging the error
          return res.status(500).json({ error: "Database error while updating user profile" });
        }

        if (results.affectedRows === 0) {
          return res.status(404).json({ error: "User not found" });
        }

        // Respond with a success message and the profile image URL
        res.json({
          message: "Profile updated successfully",
          profileImage: profileImage || "No image uploaded", // Ensure null image is handled correctly
        });
      });
    });
  }
);


// Helper function to format the birthday for SQL (YYYY-MM-DD)
function formatDateForSQL(dateString) {
  const dateObj = new Date(dateString);
  const year = dateObj.getFullYear();
  const month = String(dateObj.getMonth() + 1).padStart(2, '0'); // Ensure 2 digits
  const day = String(dateObj.getDate()).padStart(2, '0'); // Ensure 2 digits
  return `${year}-${month}-${day}`;
}


// API endpoint to follow or unfollow another user
app.post("/api/users/:userId/follow/:followingId", verifyToken, (req, res) => {
  const userId = req.params.userId;
  const followingId = req.params.followingId;

  // Ensure that the user making the request is the same as the one being followed or unfollowed
  if (req.userId.toString() !== userId) {
    return res
      .status(403)
      .json({
        error: "You are not authorized to follow or unfollow this user",
      });
  }

  // Check if the following user exists
  const checkFollowingSql = "SELECT * FROM users WHERE id = ?";
  pool.query(checkFollowingSql, [followingId], (error, followingResults) => {
    if (error) {
      return res
        .status(500)
        .json({ error: "Database error while checking following user" });
    }
    if (followingResults.length === 0) {
      return res.status(404).json({ error: "User to follow not found" });
    }

    // Check if the user is already following the other user
    const checkFollowSql =
      "SELECT * FROM follower_following WHERE follower_id = ? AND following_id = ?";
    pool.query(
      checkFollowSql,
      [userId, followingId],
      (error, followResults) => {
        if (error) {
          return res
            .status(500)
            .json({ error: "Database error while checking follow status" });
        }

        if (followResults.length > 0) {
          // User is already following, so unfollow
          const unfollowSql =
            "DELETE FROM follower_following WHERE follower_id = ? AND following_id = ?";
          pool.query(unfollowSql, [userId, followingId], (error) => {
            if (error) {
              return res
                .status(500)
                .json({ error: "Database error while unfollowing user" });
            }
            return res
              .status(200)
              .json({ message: "Unfollowed user successfully" });
          });
        } else {
          // User is not following, so follow
          const followSql =
            "INSERT INTO follower_following (follower_id, following_id) VALUES (?, ?)";
          pool.query(followSql, [userId, followingId], (error) => {
            if (error) {
              return res
                .status(500)
                .json({ error: "Database error while following user" });
            }
            return res
              .status(201)
              .json({ message: "Followed user successfully" });
          });
        }
      }
    );
  });
});


// API endpoint to check follow status of a user
app.get("/api/users/:userId/follow/:followingId/status", verifyToken, (req, res) => {
  const userId = req.params.userId;
  const followingId = req.params.followingId;

  // Ensure that the user making the request is the same as the one being checked
  if (req.userId.toString() !== userId) {
      return res
          .status(403)
          .json({ error: "You are not authorized to check follow status for this user" });
  }

  // Check if the following user exists
  const checkFollowingSql = "SELECT * FROM users WHERE id = ?";
  pool.query(checkFollowingSql, [followingId], (error, followingResults) => {
      if (error) {
          return res.status(500).json({ error: "Database error while checking following user" });
      }
      if (followingResults.length === 0) {
          return res.status(404).json({ error: "User to check follow status not found" });
      }

      // Check if the user is already following the other user
      const checkFollowSql = "SELECT * FROM follower_following WHERE follower_id = ? AND following_id = ?";
      pool.query(
          checkFollowSql,
          [userId, followingId],
          (error, followResults) => {
              if (error) {
                  return res.status(500).json({ error: "Database error while checking follow status" });
              }

              // If the user is following, return true, else return false
              const isFollowing = followResults.length > 0;
              return res.status(200).json({ isFollowing });
          }
      );
  });
});


// api comment
app.post("/api/posts/:postId/comment", verifyToken, (req, res) => {
  try {
    const { postId } = req.params; // ดึง postId จากพารามิเตอร์
    const { content } = req.body; // ดึงเนื้อหาคอมเมนต์จาก Body
    const userId = req.userId; // ดึง userId จาก Token ที่ผ่านการตรวจสอบแล้ว

    // ตรวจสอบว่าเนื้อหาคอมเมนต์ไม่ว่างเปล่า
    if (!content || content.trim() === "") {
      return res.status(400).json({ error: "Comment content cannot be empty" });
    }

    // SQL สำหรับการเพิ่มคอมเมนต์ใหม่ลงในฐานข้อมูล
    const insertCommentSql = `
      INSERT INTO comments (post_id, user_id, comment_text)
      VALUES (?, ?, ?);
    `;

    pool.query(
      insertCommentSql,
      [postId, userId, content],
      (error, results) => {
        if (error) {
          console.error("Database error during comment insertion:", error);
          return res
            .status(500)
            .json({ error: "Error saving comment to the database" });
        }

        res.status(201).json({
          message: "Comment added successfully",
          comment_id: results.insertId,
          post_id: postId,
          user_id: userId,
          content,
        });
      }
    );
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// DELETE /posts/:postId/comment/:commentId
app.delete("/api/posts/:postId/comment/:commentId", verifyToken, (req, res) => {
  const { postId, commentId } = req.params; // Extract postId and commentId from URL parameters
  const userId = req.userId; // Extract userId from the verified token

  // SQL Query to check if the comment exists and belongs to the user
  const checkCommentSql = "SELECT * FROM comments WHERE id = ? AND user_id = ? AND post_id = ?";

  pool.query(checkCommentSql, [commentId, userId, postId], (err, results) => {
    if (err) {
      console.error("Database error during comment check:", err);
      return res.status(500).json({ error: "Error checking comment" });
    }

    if (results.length === 0) {
      return res.status(404).json({ error: "Comment not found or you are not authorized to delete this comment" });
    }

    // SQL Query to delete the comment
    const deleteCommentSql = "DELETE FROM comments WHERE id = ? AND user_id = ? AND post_id = ?";

    pool.query(deleteCommentSql, [commentId, userId, postId], (err, results) => {
      if (err) {
        console.error("Database error during comment deletion:", err);
        return res.status(500).json({ error: "Error deleting comment" });
      }

      if (results.affectedRows === 0) {
        return res.status(404).json({ error: "Comment not found or not deleted" });
      }

      // After successfully deleting the comment, delete related notification for that specific comment
      const deleteNotificationSql = "DELETE FROM notifications WHERE comment_id = ?";

      pool.query(deleteNotificationSql, [commentId], (err, notificationResults) => {
        if (err) {
          console.error("Database error during notification deletion:", err);
          return res.status(500).json({ error: "Error deleting notifications" });
        }

        return res.status(200).json({ message: "Comment and associated notification deleted successfully" });
      });
    });
  });
});


//bookmark
app.post("/api/posts/:postId/bookmark", verifyToken, (req, res) => {
  const { postId } = req.params; // Extract postId from URL parameters
  const userId = req.userId; // Extract userId from the verified token

  // Check if postId is provided
  if (!postId) {
    return res.status(400).json({ error: "Post ID is required" });
  }

  // SQL Query to check if the post is already bookmarked by the user
  const checkBookmarkSql = "SELECT * FROM bookmarks WHERE user_id = ? AND post_id = ?";

  pool.query(checkBookmarkSql, [userId, postId], (err, results) => {
    if (err) {
      console.error("Database error during checking bookmark status:", err);
      return res.status(500).json({ error: "Error checking bookmark status" });
    }

    if (results.length > 0) {
      // If the post is already bookmarked, remove it (unbookmark)
      const removeBookmarkSql = "DELETE FROM bookmarks WHERE user_id = ? AND post_id = ?";
      pool.query(removeBookmarkSql, [userId, postId], (err, deleteResults) => {
        if (err) {
          console.error("Database error during removing bookmark:", err);
          return res.status(500).json({ error: "Error removing bookmark" });
        }
        return res.status(200).json({ message: "Post removed from bookmarks successfully" });
      });
    } else {
      // If the post is not bookmarked, add it
      const addBookmarkSql = "INSERT INTO bookmarks (user_id, post_id) VALUES (?, ?)";
      pool.query(addBookmarkSql, [userId, postId], (err, insertResults) => {
        if (err) {
          console.error("Database error during adding bookmark:", err);
          return res.status(500).json({ error: "Error adding post to bookmarks" });
        }
        return res.status(201).json({ message: "Post added to bookmarks successfully" });
      });
    }
  });
});


// API for fetching user's bookmarked posts
app.get("/api/bookmarks", verifyToken, (req, res) => {
  const user_id = req.userId; // Get user_id from token

  // SQL query to fetch bookmarked posts with like and comment counts and follow status
  const fetchBookmarksSql = `
    SELECT 
      p.id AS post_id, 
      p.title,
      p.content, 
      p.photo_url, 
      p.video_url, 
      p.updated_at,
      (SELECT COUNT(*) FROM likes WHERE post_id = p.id) AS like_count,
      (SELECT COUNT(*) FROM comments WHERE post_id = p.id) AS comment_count,
      u.id AS user_id, 
      u.username AS author_username, 
      u.picture AS author_profile_image,
      CASE 
        WHEN (SELECT COUNT(*) 
              FROM follower_following 
              WHERE follower_id = ? AND following_id = u.id) > 0 
        THEN TRUE ELSE FALSE 
      END AS is_following 
FROM bookmarks b
JOIN posts p ON b.post_id = p.id
JOIN users u ON p.user_id = u.id
WHERE b.user_id = ? 
  AND p.status = 'active'
ORDER BY b.created_at DESC;

  `;

  pool.query(fetchBookmarksSql, [user_id, user_id], (err, results) => {
    if (err) {
      console.error("Database error during fetching bookmarks:", err);
      return res.status(500).json({ error: "Error fetching bookmarks" });
    }

    if (results.length === 0) {
      return res.status(404).json({ message: "No bookmarks found." });
    }

    // Process photo_url and video_url as JSON arrays and format the response
    const formattedBookmarks = results.map((post) => {
      let photos = [];
      let videos = [];

      // Handle photo_url (if it's a JSON string, parse it into an array)
      if (typeof post.photo_url === "string") {
        try {
          photos = JSON.parse(post.photo_url);
        } catch (e) {
          console.error("Error parsing photo_url:", e.message);
        }
      } else if (Array.isArray(post.photo_url)) {
        photos = post.photo_url;
      }

      // Handle video_url (if it's a JSON string, parse it into an array)
      if (typeof post.video_url === "string") {
        try {
          videos = JSON.parse(post.video_url);
        } catch (e) {
          console.error("Error parsing video_url:", e.message);
        }
      } else if (Array.isArray(post.video_url)) {
        videos = post.video_url;
      }

      return {
        post_id: post.post_id,
        title: post.title,
        content: post.content,
        created_at: post.updated_at,
        like_count: post.like_count,
        comment_count: post.comment_count,
        photos, // formatted photos array
        videos, // formatted videos array
        author: {
          user_id: post.user_id,
          username: post.author_username,
          profile_image: post.author_profile_image,
        },
        is_following: post.is_following === 1, // Convert 1 to true and 0 to false
      };
    });

    // Return the formatted bookmarks
    res.json({ bookmarks: formattedBookmarks });
  });
});


//########################################################   Notification API  #######################################################


app.post("/api/notifications", verifyToken, (req, res) => {
  const { user_id, post_id, action_type, content, comment_id } = req.body;

  if (!user_id || !action_type) {
    return res.status(400).json({ error: "Missing required fields: user_id or action_type" });
  }

  // ตรวจสอบว่าเป็น action_type อะไร
  if (action_type === 'comment') {
    // สำหรับ comment ให้สร้าง Notification ใหม่ทุกครั้ง
    const insertNotificationSql = `
      INSERT INTO notifications (user_id, post_id, comment_id, action_type, content)
      VALUES (?, ?, ?, ?, ?);
    `;
    const values = [user_id, post_id || null, comment_id || null, action_type, content || null];

    pool.query(insertNotificationSql, values, (error, results) => {
      if (error) {
        console.error("Database error during notification creation:", error);
        return res.status(500).json({ error: "Error creating notification" });
      }
      res.status(201).json({
        message: "Notification created successfully",
        notification_id: results.insertId,
      });
    });
  } else {
    // สำหรับ like หรือ follow ให้ตรวจสอบ Notification เดิมก่อน
    const checkNotificationSql = `
      SELECT id FROM notifications 
      WHERE user_id = ? AND post_id = ? AND action_type = ?;
    `;
    const checkValues = [user_id, post_id || null, action_type];

    pool.query(checkNotificationSql, checkValues, (checkError, checkResults) => {
      if (checkError) {
        console.error("Database error during notification checking:", checkError);
        return res.status(500).json({ error: "Error checking notification" });
      }

      // ถ้าพบ Notification เดิม
      if (checkResults.length > 0) {
        const existingNotificationId = checkResults[0].id;

        // ถ้าเป็น `like` หรือ `follow` ซ้ำ ให้ลบ Notification เดิม
        if (action_type === 'like' || action_type === 'follow') {
          const deleteNotificationSql = `DELETE FROM notifications WHERE id = ?`;
          pool.query(deleteNotificationSql, [existingNotificationId], (deleteError) => {
            if (deleteError) {
              console.error("Database error during notification deletion:", deleteError);
              return res.status(500).json({ error: "Error deleting notification" });
            }
            return res.status(200).json({ message: `${action_type} notification removed successfully` });
          });
        } else {
          return res.status(200).json({ message: "Notification already exists" });
        }
      } else {
        // ถ้าไม่มี Notification เดิม ให้เพิ่ม Notification ใหม่
        const insertNotificationSql = `
          INSERT INTO notifications (user_id, post_id, comment_id, action_type, content)
          VALUES (?, ?, ?, ?, ?);
        `;
        const values = [user_id, post_id || null, comment_id || null, action_type, content || null];

        pool.query(insertNotificationSql, values, (error, results) => {
          if (error) {
            console.error("Database error during notification creation:", error);
            return res.status(500).json({ error: "Error creating notification" });
          }
          res.status(201).json({
            message: "Notification created successfully",
            notification_id: results.insertId,
          });
        });
      }
    });
  }
});


app.get("/api/notifications", verifyToken, (req, res) => {
  const userId = req.userId;

  const fetchActionNotificationsSql = `
  SELECT 
    n.id, 
    n.user_id AS receiver_id, 
    n.post_id, 
    n.comment_id,   
    n.action_type, 
    n.content, 
    n.read_status,
    n.created_at,
    n.ads_id, -- เพิ่มตรงนี้
    s.username AS sender_name,
    s.picture AS sender_picture, 
    p_owner.username AS receiver_name,
    c.comment_text AS comment_content  
  FROM notifications n
  LEFT JOIN users s ON n.user_id = s.id
  LEFT JOIN posts p ON n.post_id = p.id
  LEFT JOIN users p_owner ON p.user_id = p_owner.id
  LEFT JOIN comments c ON n.post_id = c.post_id AND n.action_type = 'comment' 
  WHERE n.action_type IN ('comment', 'like', 'follow', 'ads_status_change')
    AND (
      (n.action_type = 'ads_status_change' AND n.user_id = ?) -- สำหรับ noti โฆษณา
      OR
      (n.action_type IN ('comment', 'like', 'follow') AND p_owner.id = ?) -- สำหรับ noti โพสต์
    )
  ORDER BY n.created_at DESC;
  `;

  pool.query(fetchActionNotificationsSql, [userId, userId], (error, results) => {
    if (error) {
      console.error("Database error during fetching notifications:", error);
      return res.status(500).json({ error: "Error fetching notifications" });
    }
    res.json(results);
  });
});


// API สำหรับอัปเดตสถานะการอ่านของ Notification ตาม ID
app.put("/api/notifications/:id/read", verifyToken, (req, res) => {
  const { id } = req.params;  // รับ notification ID จาก URL พารามิเตอร์
  const userId = req.userId;  // รับ userId ที่ได้จาก verifyToken middleware

  // log ค่า ID และ userId สำหรับการดีบัก
  console.log("Notification ID:", id);
  console.log("User ID from Token (Post Owner):", userId);

  // คำสั่ง SQL สำหรับการอัปเดตสถานะการอ่านของ Notification โดยตรวจสอบว่า userId คือเจ้าของโพสต์
  const updateReadStatusSql = `
    UPDATE notifications n
    JOIN posts p ON n.post_id = p.id
    SET n.read_status = 1
    WHERE n.id = ? AND p.user_id = ?;
  `;

  // เรียกคำสั่ง SQL
  pool.query(updateReadStatusSql, [id, userId], (error, results) => {
    if (error) {
      // หากเกิดข้อผิดพลาดในการทำงานกับฐานข้อมูล
      console.error("Database error during updating read status:", error);
      return res.status(500).json({ error: "Error updating read status" });
    }
    
    // ตรวจสอบว่ามีการอัปเดตหรือไม่
    if (results.affectedRows === 0) {
      // log กรณีไม่พบ notification หรือตรวจสอบว่า user ไม่ใช่เจ้าของโพสต์
      console.warn(`Notification not found or you are not the owner of the post (User ID: ${userId})`);
      return res.status(404).json({ message: "Notification not found or you are not the owner of the post" });
    }

    // หากอัปเดตสำเร็จ
    console.log("Notification marked as read for ID:", id);
    res.json({ message: "Notification marked as read" });
  });
});


// API สำหรับลบ Notification
app.delete("/api/notifications", verifyToken, (req, res) => {
  const { user_id, post_id, action_type } = req.body;

  if (!user_id || !post_id || !action_type) {
    return res
      .status(400)
      .json({ error: "Missing required fields: user_id, post_id, or action_type" });
  }

  const deleteNotificationSql = `
    DELETE FROM notifications 
    WHERE user_id = ? AND post_id = ? AND action_type = ?;
  `;

  pool.query(deleteNotificationSql, [user_id, post_id, action_type], (error, results) => {
    if (error) {
      console.error("Database error during deleting notification:", error);
      return res.status(500).json({ error: "Error deleting notification" });
    }
    res.json({ message: "Notification deleted successfully" });
  });
});


app.post("/api/ads/:id/notify-status-change", verifyToken, (req, res) => {
  const adId = req.params.id;
  const { new_status, admin_notes } = req.body;

  // ดึงข้อมูล ads
  const getAdSql = `SELECT * FROM ads WHERE id = ?`;
  pool.query(getAdSql, [adId], (adErr, adResults) => {
    if (adErr || adResults.length === 0) {
      return res.status(404).json({ error: "ไม่พบโฆษณานี้" });
    }
    const ad = adResults[0];

    // สร้างข้อความแจ้งเตือน
    let content = `โฆษณาของคุณ (${ad.title}) มีการเปลี่ยนสถานะเป็น "${new_status}"`;
    if (new_status === "rejected" && admin_notes) {
      content += `\nเหตุผลที่ถูกปฏิเสธ: ${admin_notes}`;
    }

    // เพิ่ม notification
    const insertNotificationSql = `
      INSERT INTO notifications (user_id, action_type, content)
      VALUES (?, ?, ?)
    `;
    pool.query(
      insertNotificationSql,
      [ad.user_id, "ads_status_change", content],
      (notiErr, notiResults) => {
        if (notiErr) {
          return res.status(500).json({ error: "บันทึกแจ้งเตือนไม่สำเร็จ" });
        }
        res.status(201).json({ message: "แจ้งเตือนสถานะโฆษณาสำเร็จ" });
      }
    );
  });
});


// ฟังก์ชันช่วยจัดการการอัปเดต Order และ Ad เพื่อลดความซับซ้อน
function proceedUpdateOrderAndAd(connection, orderId, slipImagePath, renewAdsId, packageId, originalAdExpirationDate, originalAdShowAt, canUpload, res) {
  if (!canUpload) {
      return connection.rollback(() => {
          connection.release();
          res.status(400).json({ error: 'ไม่สามารถอัปโหลดสลิปได้ เนื่องจากสถานะไม่ถูกต้อง' });
      });
  }

  const updateOrderSql = 'UPDATE orders SET slip_image = ?, status = "paid", updated_at = NOW() WHERE id = ?';
  connection.query(updateOrderSql, [slipImagePath, orderId], (updateOrderErr, updateOrderResult) => {
      if (updateOrderErr) {
          return connection.rollback(() => {
              connection.release();
              console.error(`[ERROR] Database error updating slip_image for order ${orderId}:`, updateOrderErr);
              res.status(500).json({ error: 'เกิดข้อผิดพลาดในการบันทึกสลิป' });
          });
      }

      if (renewAdsId !== null) {
          const getDurationSql = 'SELECT duration_days FROM ad_packages WHERE package_id = ?';
          connection.query(getDurationSql, [packageId], (durationErr, durationResults) => {
              if (durationErr || durationResults.length === 0) {
                  console.error(`[ERROR] Failed to get duration for package ${packageId} on order ${orderId}:`, durationErr || 'No package info found');
                  return connection.commit(() => {
                      connection.release();
                      res.json({ message: 'อัปโหลดสลิปสำเร็จ แต่มีปัญหาในการต่ออายุโฆษณา กรุณาติดต่อแอดมิน', slip_path: slipImagePath });
                  });
              }
              const duration_days = durationResults[0].duration_days;

              // คำนวณ expiration_date ใหม่โดยบวกจาก originalAdExpirationDate
              const newExpirationDate = new Date(originalAdExpirationDate);
              newExpirationDate.setDate(newExpirationDate.getDate() + duration_days);

              const updateAdsSql = `
                  UPDATE ads
                  SET status = 'active',
                      -- ไม่ต้องอัปเดต show_at ถ้าเป็นการต่ออายุ เพราะ show_at คือวันเริ่มแสดงครั้งแรก
                      expiration_date = ?,
                      updated_at = NOW()
                  WHERE id = ?;
              `;
              // ใช้ newExpirationDate ในการอัปเดต expiration_date
              connection.query(updateAdsSql, [newExpirationDate, renewAdsId], (updateAdsErr) => {
                  if (updateAdsErr) {
                      console.error(`[ERROR] Database error updating ad ${renewAdsId} for order ${orderId}:`, updateAdsErr);
                      return connection.commit(() => {
                          connection.release();
                          res.json({ message: 'อัปโหลดสลิปสำเร็จ แต่มีปัญหาในการต่ออายุโฆษณา กรุณาติดต่อแอดมิน', slip_path: slipImagePath });
                      });
                  }
                  console.log(`[INFO] Ad ${renewAdsId} successfully renewed and set to 'active' via order ${orderId}.`);
                  console.log(`[INFO] โฆษณาของคุณได้รับการต่ออายุเพิ่มอีก ${duration_days} วันแล้วงับ!`); // เพิ่ม log ตรงนี้

                  // เพิ่มการแจ้งเตือนหลังต่ออายุสำเร็จ (ข้อความใหม่)
                  const notiMsg = `โฆษณาของคุณได้รับการต่ออายุเพิ่มอีก ${duration_days} วัน สำเร็จแล้ว`;
                  connection.query(
                    `SELECT user_id FROM ads WHERE id = ?`,
                    [renewAdsId],
                    (err, results) => {
                        if (!err && results.length > 0) {
                            const user_id = results[0].user_id;
                            connection.query(
                                `INSERT INTO notifications (user_id, post_id, ads_id, action_type, content, created_at, read_status, comment_id) VALUES (?, NULL, ?, 'ads_status_change', ?, NOW(), 0, NULL)`,
                                [user_id, renewAdsId, notiMsg],
                                (notiErr) => {
                                    if (notiErr) {
                                        console.error(`[ERROR] Failed to send notification for renewed ad ID ${renewAdsId}:`, notiErr);
                                    } else {
                                        console.log(`[INFO] Sent notification for renewed ad ID ${renewAdsId}`);
                                    }
                                    connection.commit(() => {
                                        connection.release();
                                        res.json({ message: `อัปโหลดสลิปสำเร็จ! โฆษณาของคุณได้รับการต่ออายุเพิ่มอีก ${duration_days} วันเรียบร้อยแล้ว`, slip_path: slipImagePath });
                                    });
                                }
                            );
                        } else {
                            connection.commit(() => {
                                connection.release();
                                res.json({ message: `อัปโหลดสลิปสำเร็จ! โฆษณาของคุณได้รับการต่ออายุเพิ่มอีก ${duration_days} วันเรียบร้อยแล้ว`, slip_path: slipImagePath });
                            });
                        }
                    }
                  );
              });
          });

      } else {
          // โฆษณาใหม่
          const updateAdsSql = 'UPDATE ads SET status = "paid", updated_at = NOW() WHERE order_id = ?';
          connection.query(updateAdsSql, [orderId], (updateAdsErr2) => {
              if (updateAdsErr2) {
                  console.error(`[ERROR] Database error updating ads status for order ${orderId}:`, updateAdsErr2);
              }
              console.log(`[INFO] Ad for Order ID ${orderId} set to 'paid'.`);

              connection.query('SELECT id FROM ads WHERE order_id = ?', [orderId], (getAdIdErr, adIdResults) => {
                  if (getAdIdErr || adIdResults.length === 0) {
                      console.error(`[ERROR] Could not find ad_id for order ${orderId} to send notification:`, getAdIdErr);
                  } else {
                      notifyAdsStatusChange(adIdResults[0].id, 'paid', null, (notiErr) => {
                          if (notiErr) {
                              console.error(`[ERROR] Failed to send paid notification for new ad order ${orderId}:`, notiErr);
                          } else {
                              console.log(`[INFO] Sent paid notification for new ad order ${orderId}`);
                          }
                      });
                  }
                  connection.commit(() => {
                      connection.release();
                      res.json({ message: 'อัปโหลดสลิปสำเร็จ! กรุณารอแอดมินตรวจสอบสลิป', slip_path: slipImagePath });
                  });
              });
          });
      }
  });
}


// ฟังก์ชันสำหรับจัดรูปแบบวันที่ให้เป็นภาษาไทยและปีพุทธศักราช
function formatThaiDate(dateString) {
  const date = new Date(dateString);
  // ใช้ Intl.DateTimeFormat เพื่อจัดรูปแบบเป็นวันที่ภาษาไทยและปีพุทธศักราช
  const formatter = new Intl.DateTimeFormat('th-TH', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      calendar: 'buddhist' // ใช้ปีพุทธศักราช
  });
  return formatter.format(date);
}

// ฟังก์ชันสำหรับสร้าง notification เมื่อ ads เปลี่ยนสถานะ
function notifyAdsStatusChange(adId, newStatus, adminNotes = null, callback) {
  // ขั้นตอนที่ 1: ดึง user_id และ expiration_date ล่าสุดจากตาราง ads โดยใช้ adId
  pool.query('SELECT user_id, expiration_date FROM ads WHERE id = ?', [adId], (err, adsResults) => {
      if (err || adsResults.length === 0) {
          // หากเกิดข้อผิดพลาดในการดึงข้อมูลหรือหา ad ไม่พบ ก็ส่ง callback กลับไปพร้อมข้อผิดพลาด
          return callback(err || new Error('Ad not found'));
      }
      const { user_id, expiration_date } = adsResults[0]; // ดึง user_id และ expiration_date ออกมา
      let content = ''; // เตรียมตัวแปรสำหรับเก็บข้อความแจ้งเตือน

      // ขั้นตอนที่ 2: ตรวจสอบว่าเป็น "การต่ออายุที่ชำระเงินแล้ว" หรือไม่
      // โดยการ query ตาราง orders เพื่อดูว่ามีรายการที่ renew_ads_id ตรงกับ adId
      // และมี status เป็น 'paid' และดึง package_id มาด้วย
      pool.query(
          `SELECT package_id FROM orders WHERE renew_ads_id = ? AND status = 'paid'`,
          [adId],
          (err, orderResults) => {
              if (err) {
                  // หากเกิดข้อผิดพลาดในการดึงข้อมูลจากตาราง orders
                  return callback(err);
              }

              // เงื่อนไขหลักในการกำหนดข้อความแจ้งเตือน
              // ถ้าสถานะใหม่คือ 'active' และพบว่ามีการต่ออายุที่ชำระเงินแล้วในตาราง orders
              if (newStatus === 'active' && orderResults.length > 0) {
                  const renewedPackageId = orderResults[0].package_id;

                  // ขั้นตอนที่ 3: ดึง duration_days จากตาราง ad_packages โดยใช้ package_id ที่ได้จาก orders
                  pool.query(
                      `SELECT duration_days FROM ad_packages WHERE package_id = ?`,
                      [renewedPackageId],
                      (err, packageResults) => {
                          if (err) {
                              return callback(err);
                          }

                          let renewedDays = 'ไม่ระบุ'; // ค่าเริ่มต้นถ้าหา duration_days ไม่เจอ
                          if (packageResults.length > 0) {
                              renewedDays = packageResults[0].duration_days;
                          }

                          // จัดรูปแบบวันที่หมดอายุใหม่ให้เป็นภาษาไทยและปีพุทธศักราช
                          const formattedExpirationDate = formatThaiDate(expiration_date);

                          // สร้างข้อความแจ้งเตือนการต่ออายุที่ละเอียดขึ้น
                          content = `โฆษณาของคุณได้รับการต่ออายุ ${renewedDays} วันสำเร็จแล้ว โฆษณานี้ขยายหมดอายุเป็นวันที่ ${formattedExpirationDate}`;

                          // ขั้นตอนสุดท้าย: บันทึกข้อมูลการแจ้งเตือนลงในตาราง notifications
                          pool.query(
                              `INSERT INTO notifications (user_id, action_type, content, ads_id) VALUES (?, 'ads_status_change', ?, ?)`,
                              [user_id, content, adId],
                              callback
                          );
                      }
                  );
              } else {
                  // ถ้าไม่ใช่กรณีของการต่ออายุที่ชำระเงินแล้ว หรือสถานะไม่ใช่ 'active'
                  // ให้ใช้ switch case เดิม เพื่อกำหนดข้อความตามสถานะปกติ
                  switch (newStatus) {
                      case 'approved':
                          content = 'โฆษณาของคุณได้รับการตรวจสอบแล้ว กรุณาโอนเงินเพื่อแสดงโฆษณา';
                          break;
                      case 'active':
                          // ข้อความนี้จะใช้เฉพาะกรณีที่ 'active' แต่ไม่ใช่การต่ออายุครั้งแรก
                          content = 'โฆษณาของคุณได้รับการอนุมัติขึ้นแสดงแล้ว';
                          break;
                      case 'rejected':
                          content = `โฆษณาของคุณถูกปฏิเสธ เหตุผล: ${adminNotes || '-'}`;
                          break;
                      case 'paid':
                          content = 'โฆษณาของคุณชำระเงินเรียบร้อยแล้ว รอแอดมินตรวจสอบ';
                          break;
                      case 'expired':
                          content = 'โฆษณาของคุณหมดอายุแล้ว';
                          break;
                      case 'expiring_soon':
                          content = 'โฆษณาของคุณจะหมดอายุในอีก 3 วัน กรุณาต่ออายุเพื่อการแสดงผลอย่างต่อเนื่อง';
                          break;
                      default:
                          content = `สถานะโฆษณาของคุณเปลี่ยนเป็น ${newStatus}`;
                  }

                  // ขั้นตอนสุดท้าย: บันทึกข้อมูลการแจ้งเตือนลงในตาราง notifications
                  pool.query(
                      `INSERT INTO notifications (user_id, action_type, content, ads_id) VALUES (?, 'ads_status_change', ?, ?)`,
                      [user_id, content, adId],
                      callback
                  );
              }
          }
      );
  });
}


// ตัวอย่างฟังก์ชันสำหรับตรวจสอบโฆษณาที่กำลังจะหมดอายุ (ต้องรันเป็น Cron Job)
function checkExpiringAds() {
    console.log('[INFO] Checking for expiring ads...');
    // เลือกโฆษณาที่มีสถานะ 'active' และ expiration_date เหลืออีก 3 วัน
    const sql = `
        SELECT id, user_id, expiration_date
        FROM ads
        WHERE status = 'active'
        AND expiration_date = CURDATE() + INTERVAL 3 DAY;
    `;
    pool.query(sql, (err, results) => {
        if (err) {
            console.error('[ERROR] Database error checking expiring ads:', err);
            return;
        }
        if (results.length > 0) {
            console.log(`[INFO] Found ${results.length} ads expiring soon.`);
            results.forEach(ad => {
                // เรียกฟังก์ชันแจ้งเตือน
                notifyAdsStatusChange(ad.id, 'expiring_soon', null, (notiErr) => {
                    if (notiErr) {
                        console.error(`[ERROR] Failed to send expiring_soon notification for ad ID ${ad.id}:`, notiErr);
                    } else {
                        console.log(`[INFO] Sent expiring_soon notification for ad ID ${ad.id}`);
                    }
                });
            });
        } else {
            console.log('[INFO] No ads expiring soon today.');
        }
    });
}


//########################################################   Bookmark API  #######################################################


// API สำหรับเพิ่มบุ๊คมาร์ค
app.post("/api/bookmarks", verifyToken, (req, res) => {
  const { post_id } = req.body; // ดึง post_id จาก request body
  const user_id = req.userId; // ดึง user_id จาก Token ที่ผ่านการตรวจสอบแล้ว

  // ตรวจสอบว่ามี post_id ที่ต้องการบุ๊คมาร์คหรือไม่
  if (!post_id) {
    return res.status(400).json({ error: "Post ID is required" });
  }

  // เพิ่มข้อมูลบุ๊คมาร์คในฐานข้อมูล
  const addBookmarkSql = "INSERT INTO bookmarks (user_id, post_id) VALUES (?, ?)";
  pool.query(addBookmarkSql, [user_id, post_id], (err, results) => {
    if (err) {
      console.error("Database error during adding bookmark:", err);
      return res.status(500).json({ error: "Error adding bookmark" });
    }

    res.status(201).json({ message: "Post bookmarked successfully" });
  });
});


// API สำหรับลบบุ๊คมาร์ค
app.delete("/api/bookmarks", verifyToken, (req, res) => {
  const { post_id } = req.body; // ดึง post_id จาก request body
  const user_id = req.userId; // ดึง user_id จาก Token ที่ผ่านการตรวจสอบแล้ว

  // ตรวจสอบว่ามี post_id ที่ต้องการลบหรือไม่
  if (!post_id) {
    return res.status(400).json({ error: "Post ID is required" });
  }

  // ลบข้อมูลบุ๊คมาร์คจากฐานข้อมูล
  const deleteBookmarkSql = "DELETE FROM bookmarks WHERE user_id = ? AND post_id = ?";
  pool.query(deleteBookmarkSql, [user_id, post_id], (err, results) => {
    if (err) {
      console.error("Database error during deleting bookmark:", err);
      return res.status(500).json({ error: "Error deleting bookmark" });
    }

    if (results.affectedRows === 0) {
      return res.status(404).json({ message: "Bookmark not found or you are not authorized to delete" });
    }

    res.json({ message: "Bookmark deleted successfully" });
  });
});


// API สำหรับดึงรายการบุ๊คมาร์คของผู้ใช้
app.get("/api/bookmarks", verifyToken, (req, res) => {
  const user_id = req.userId; // ดึง user_id จาก Token ที่ผ่านการตรวจสอบแล้ว

  // ดึงรายการบุ๊คมาร์คจากฐานข้อมูล
  const fetchBookmarksSql = `
    SELECT 
      b.post_id, 
      p.title, 
      p.content, 
      p.photo_url, 
      p.video_url, 
      u.username AS author 
    FROM bookmarks b
    JOIN posts p ON b.post_id = p.id
    JOIN users u ON p.user_id = u.id
    WHERE b.user_id = ?
    ORDER BY b.created_at DESC;
  `;

  pool.query(fetchBookmarksSql, [user_id], (err, results) => {
    if (err) {
      console.error("Database error during fetching bookmarks:", err);
      return res.status(500).json({ error: "Error fetching bookmarks" });
    }

    res.json(results);
  });
});

//check bookmark status
app.get("/api/posts/:postId/bookmark/status", verifyToken, (req, res) => {
  const { postId } = req.params; // Extract postId from URL parameters
  const userId = req.userId; // Extract userId from the verified token

  // SQL Query to check if the post is already bookmarked by the user
  const checkBookmarkSql = "SELECT * FROM bookmarks WHERE user_id = ? AND post_id = ?";

  pool.query(checkBookmarkSql, [userId, postId], (err, results) => {
    if (err) {
      console.error("Database error during checking bookmark status:", err);
      return res.status(500).json({ error: "Error checking bookmark status" });
    }

    // If results are found, post is bookmarked, otherwise it is not
    const isBookmarked = results.length > 0;
    res.status(200).json({ isBookmarked });
  });
});


// API to get posts from followed users
app.get("/api/following/posts", verifyToken, (req, res) => {
  const userId = req.userId; // The logged-in user who is following others

  const getFollowedPostsSql = `
    SELECT 
      p.id AS id, 
      p.user_id AS userId, 
      p.title AS title, 
      p.content AS content, 
      p.photo_url AS photoUrl, 
      p.video_url AS videoUrl, 
      u.username AS userName, 
      u.picture AS userProfileUrl, 
      p.updated_at AS updated, 
      (SELECT COUNT(*) FROM likes WHERE post_id = p.id) AS likeCount, 
      (SELECT COUNT(*) FROM comments WHERE post_id = p.id) AS commentCount, 
      EXISTS(SELECT 1 FROM likes WHERE post_id = p.id AND user_id = ?) AS is_liked
    FROM posts p
    JOIN follower_following f ON p.user_id = f.following_id
    JOIN users u ON p.user_id = u.id
    WHERE f.follower_id = ?
    ORDER BY p.updated_at DESC;
  `;

  pool.query(getFollowedPostsSql, [userId, userId], (error, results) => {
    if (error) {
      return res.status(500).json({ error: "Database error during fetching followed posts." });
    }

    // If there are no followed posts, return an empty array
    if (results.length === 0) {
      return res.status(200).json({ message: "No posts from followed users.", posts: [] });
    }

    // Format the response and send it
    const parsedResults = results.map((post) => {
      const photoUrls = Array.isArray(post.photoUrl) ? post.photoUrl : []; // Fix: Use 'photoUrl' for consistency
      const videoUrls = Array.isArray(post.videoUrl) ? post.videoUrl : []; // Fix: Use 'videoUrl' for consistency

      return {
        id: post.id,
        userId: post.userId,
        title: post.title,
        content: post.content,
        updated: post.updated, 
        photo_url: photoUrls,
        video_url: videoUrls,
        userName: post.userName,
        userProfileUrl: post.userProfileUrl,
        likeCount: post.likeCount || 0,
        commentCount: post.commentCount || 0,
        isLiked: !!post.is_liked, // Convert to Boolean
      };
    });

    res.status(200).json({ posts: parsedResults }); // Return the parsed results
  });
});


//########################################################   Report API  ########################################################


// API for reporting a post
app.post("/api/posts/:postId/report", verifyToken, (req, res) => {
  const { postId } = req.params;
  const { reason } = req.body;
  const userId = req.userId; // Extract userId from the token

  // Validate the reason for reporting
  if (!reason || reason.trim() === "") {
    return res.status(400).json({ error: "Report reason is required" });
  }

  // SQL to insert the report into the database
  const insertReportSql = `
    INSERT INTO reports (user_id, post_id, reason)
    VALUES (?, ?, ?);
  `;

  pool.query(insertReportSql, [userId, postId, reason], (error, results) => {
    if (error) {
      console.error("Database error during reporting post:", error);
      return res.status(500).json({ error: "Error reporting post" });
    }

    res.status(201).json({ message: "Post reported successfully" });
  });
});


// API for retrieving all reported posts (admin-only)
app.get("/api/reports", verifyToken, (req, res) => {
  const role = req.role;

  // Only allow admin to view reports
  if (role !== "admin") {
    return res.status(403).json({ error: "Unauthorized access" });
  }

  const fetchReportsSql = `
    SELECT r.*, u.username AS reported_by, p.title AS post_title
    FROM reports r
    JOIN users u ON r.user_id = u.id
    JOIN posts p ON r.post_id = p.id
    WHERE r.status = 'pending'
    ORDER BY r.reported_at DESC;
  `;

  pool.query(fetchReportsSql, (error, results) => {
    if (error) {
      console.error("Database error during fetching reports:", error);
      return res.status(500).json({ error: "Error fetching reports" });
    }

    res.json(results);
  });
});


// Soft Delete a User, Hard Delete their Posts, and Delete Follows
app.delete("/api/users/:id", verifyToken, (req, res) => {
  const { id } = req.params;
  const user_id = req.userId; // Get user ID from the token

  // Only allow the user to delete their own account or admin role
  if (parseInt(user_id) !== parseInt(id) && req.role !== "admin") {
    return res.status(403).json({ error: "You do not have permission to delete this user." });
  }

  // First, delete all posts of the user (hard delete)
  const deletePostsSql = "DELETE FROM posts WHERE user_id = ?";
  pool.query(deletePostsSql, [id], (postErr, postResults) => {
    if (postErr) {
      console.error("Database error during post deletion:", postErr);
      return res.status(500).json({ error: "Database error during post deletion" });
    }

    // Next, delete all follows of the user (both following and followers)
    const deleteFollowsSql = "DELETE FROM follower_following WHERE follower_id = ? OR following_id = ?";
    pool.query(deleteFollowsSql, [id, id], (followErr, followResults) => {
      if (followErr) {
        console.error("Database error during follow deletion:", followErr);
        return res.status(500).json({ error: "Database error during follow deletion" });
      }

      // Now, soft delete the user (update status to 'deactivated')
      const softDeleteUserSql = "UPDATE users SET status = 'deactivated' WHERE id = ?";
      pool.query(softDeleteUserSql, [id], (userErr, userResults) => {
        if (userErr) {
          console.error("Database error during user soft deletion:", userErr);
          return res.status(500).json({ error: "Database error during user soft deletion" });
        }

        if (userResults.affectedRows === 0) {
          return res.status(404).json({ error: "User not found" });
        }

        res.json({
          message: "User soft-deleted, their posts and follows deleted successfully",
          deletedPostsCount: postResults.affectedRows, // Return the number of posts deleted
          deletedFollowsCount: followResults.affectedRows // Return the number of follows deleted
        });
      });
    });
  });
});


//########################################################   Follow API  ########################################################


//get following
app.get("/api/users/following/:userId", (req, res) => {
  // ดึง userId จาก request parameter
  const { userId } = req.params;

  // ตรวจสอบว่ามี userId ใน request หรือไม่
  if (!userId) {
    return res.status(400).json({ error: "User ID not provided" });
  }

  // Query SQL หรือการประมวลผลอื่น ๆ
  const getFollowingSql = `
    SELECT 
      u.id AS userId, 
      u.username, 
      u.picture AS profileImageUrl
    FROM follower_following f
    JOIN users u ON f.following_id = u.id
    WHERE f.follower_id = ?;
  `;
  pool.query(getFollowingSql, [userId], (err, results) => {
    if (err) {
      console.error("Database error during fetching following:", err);
      return res.status(500).json({ error: "Error fetching following" });
    }

    // ตรวจสอบว่ามีผลลัพธ์หรือไม่
    if (results.length === 0) {
      return res.status(404).json({ message: "No following found" });
    }

    // ส่งผลลัพธ์กลับไป
    res.json(results);
  });
});

//get followers
app.get("/api/users/followers/:userId", (req, res) => {
  // ดึง userId จาก request parameter
  const { userId } = req.params;

  // ตรวจสอบว่ามี userId ใน request หรือไม่
  if (!userId) {
    return res.status(400).json({ error: "User ID not provided" });
  }

  // Query SQL เพื่อตรวจสอบผู้ที่ติดตาม userId
  const getFollowersSql = `
    SELECT 
      u.id AS userId, 
      u.username, 
      u.picture AS profileImageUrl
    FROM follower_following f
    JOIN users u ON f.follower_id = u.id
    WHERE f.following_id = ?;
  `;

  pool.query(getFollowersSql, [userId], (err, results) => {
    if (err) {
      console.error("Database error during fetching followers:", err);
      return res.status(500).json({ error: "Error fetching followers" });
    }

    // ตรวจสอบว่ามีผลลัพธ์หรือไม่
    if (results.length === 0) {
      return res.status(404).json({ message: "No followers found" });
    }

    // ส่งผลลัพธ์กลับไป
    res.json(results);
  });
});


//search following
app.get("/api/users/search/following", verifyToken, (req, res) => {
  const { query } = req.query; // รับคำค้นหาจาก query parameter
  const followerId = req.userId; // รับ follower_id จาก token ของผู้ใช้ที่ล็อกอิน

  if (!query || query.trim() === "") {
    return res.status(400).json({ error: "Search query is required" });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`; // แปลงคำค้นหาเป็นรูปแบบ LIKE

  const searchFollowingSql = `
    SELECT 
      u.id AS userId, 
      u.username, 
      u.picture AS profileImageUrl
    FROM follower_following f
    JOIN users u ON f.following_id = u.id
    WHERE f.follower_id = ? AND LOWER(u.username) LIKE ? AND u.status = 'active';
  `;

  pool.query(searchFollowingSql, [followerId, searchValue], (err, results) => {
    if (err) {
      console.error("Database error during following search:", err.message || err);
      return res.status(500).json({ error: "Error fetching following" });
    }

    // ส่งกลับ array ว่างหากไม่พบผลลัพธ์
    res.status(200).json(results || []);
  });
});


app.get("/api/users/search/followers", verifyToken, (req, res) => {
  const { query } = req.query; // รับคำค้นหาจาก query parameter
  const followingId = req.userId; // รับ following_id จาก token ของผู้ใช้ที่ล็อกอิน (คนที่ถูกติดตาม)

  if (!query || query.trim() === "") {
    return res.status(400).json({ error: "Search query is required" });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`; // แปลงคำค้นหาเป็นรูปแบบ LIKE

  const searchFollowersSql = `
    SELECT 
      u.id AS userId, 
      u.username, 
      u.picture AS profileImageUrl
    FROM follower_following f
    JOIN users u ON f.follower_id = u.id
    WHERE f.following_id = ? AND LOWER(u.username) LIKE ? AND u.status = 'active';
  `;

  pool.query(searchFollowersSql, [followingId, searchValue], (err, results) => {
    if (err) {
      console.error("Database error during followers search:", err.message || err);
      return res.status(500).json({ error: "Error fetching followers" });
    }

    // ส่งกลับ array ว่างหากไม่พบผลลัพธ์
    res.status(200).json(results || []);
  });
});

// Endpoint to check bookmark status
app.get("/api/bookmarks/:post_id", verifyToken, (req, res) => {
  const post_id = req.params.post_id;
  const user_id = req.userId;

  const query = 'SELECT * FROM bookmarks WHERE user_id = ? AND post_id = ?';
  pool.query(query, [user_id, post_id], (err, results) => {
      if (err) {
          return res.status(500).send({ error: 'Database error' });
      }
      const isBookmarked = results.length > 0;
      res.send({ isBookmarked });
  });
});


//########################################################   Admin API  ########################################################
// Admin Login API
app.post("/api/admin/login", async (req, res) => {
  try {
      const { email, password } = req.body;

      const ipAddress = req.headers["x-forwarded-for"] || req.socket.remoteAddress; // Use req.socket.remoteAddress for more direct IP

      const sql = "SELECT id, email, password, username, picture, role, status, failed_attempts FROM users WHERE email = ? AND status = 'active' AND role = 'admin'";
      pool.query(sql, [email], (err, results) => {
          if (err) {
              console.error("Database error during admin login:", err);
              return res.status(500).json({ error: "Database error during admin login" });
          }
          if (results.length === 0) {
              return res.status(404).json({ message: "No admin user found with that email or user is inactive/not admin." });
          }

          const user = results[0];

          bcrypt.compare(password, user.password, (err, isMatch) => {
              if (err) {
                  console.error("Password comparison error:", err);
                  return res.status(500).json({ error: "Password comparison error" });
              }
              if (!isMatch) {
                  const updateFailSql =
                      "UPDATE users SET failed_attempts = failed_attempts + 1, last_failed_attempt = NOW() WHERE id = ?";
                  pool.query(updateFailSql, [user.id], (err) => {
                      if (err) console.error("Error logging failed login attempt:", err);
                  });

                  const remainingAttempts = Math.max(0, 5 - (user.failed_attempts + 1)); // Ensure it doesn't go below 0
                  let message = `Email or Password is incorrect.`;
                  if (remainingAttempts > 0) {
                      message += ` You have ${remainingAttempts} attempts left.`;
                  } else {
                      message += ` Your account might be locked.`;
                  }
                  return res.status(401).json({ message });
              }

              // Reset failed attempts after a successful login
              const resetFailSql =
                  "UPDATE users SET failed_attempts = 0, last_login = NOW(), last_login_ip = ? WHERE id = ?";
              pool.query(resetFailSql, [ipAddress, user.id], (err) => {
                  if (err) {
                      console.error("Error resetting failed attempts or updating login time:", err);
                      return res.status(500).json({ error: "Error updating login details." });
                  }

                  const token = jwt.sign({ id: user.id, role: user.role }, JWT_SECRET, { expiresIn: '1h' }); // Token expires in 1 hour

                  res.status(200).json({
                      message: "Admin authentication successful",
                      token,
                      user: {
                          id: user.id,
                          email,
                          username: user.username,
                          picture: user.picture,
                          role: user.role,
                          last_login: new Date(),
                          last_login_ip: ipAddress,
                      },
                  });
              });
          });
      });
  } catch (error) {
      console.error("Internal error during admin login:", error.message);
      res.status(500).json({ error: "Internal server error" });
  }
});


// Admin Dashboard: New Users per Day and Total Posts per Day
app.get("/api/admin/dashboard", authenticateToken, authorizeAdmin, (req, res) => {
  // Query to get new users count per day and total users
  const newUsersQuery = `
      SELECT 
          DATE(created_at) AS date, 
          COUNT(*) AS new_users 
      FROM users 
      WHERE role = 'user'
      GROUP BY DATE(created_at)
      ORDER BY DATE(created_at) DESC;
  `;

  // Query to get total posts per day and total posts
  const totalPostsQuery = `
      SELECT 
          DATE(updated_at) AS date, 
          COUNT(*) AS total_posts 
      FROM posts
      GROUP BY DATE(updated_at)
      ORDER BY DATE(updated_at) DESC;
  `;

  // Execute both queries in parallel
  pool.query(newUsersQuery, (newUsersError, newUsersResults) => {
      if (newUsersError) {
          console.error("Database error fetching new users:", newUsersError);
          return res.status(500).json({ error: "Error fetching new users data" });
      }

      pool.query(totalPostsQuery, (totalPostsError, totalPostsResults) => {
          if (totalPostsError) {
              console.error("Database error fetching total posts:", totalPostsError);
              return res.status(500).json({ error: "Error fetching total posts data" });
          }

          res.json({
              new_users_per_day: newUsersResults,
              total_posts_per_day: totalPostsResults,
          });
      });
  });
});


// Fetch All Active Ads in Random Order
app.get("/api/ads/random", (req, res) => {
  const fetchRandomAdsSql = `
      SELECT * FROM ads 
      WHERE status = 'active'
      ORDER BY RAND();
  `;

  pool.query(fetchRandomAdsSql, (err, results) => {
      if (err) {
          console.error("Database error during fetching random ads:", err);
          return res.status(500).json({ error: "Error fetching random ads" });
      }

      res.json(results);
  });
});


// Serve images from the uploads directory
app.use("/api/uploads", express.static('uploads'));


// Create an Ad (Admin only)
app.post("/api/ads", authenticateToken, authorizeAdmin, upload.single("image"), (req, res) => {
  const { title, content, link, status, expiration_date } = req.body;
  const image = req.file ? `/uploads/${req.file.filename}` : null;
  const userId = req.user.id; // ดึง user_id จาก token ที่ authenticateToken ใส่ไว้

  if (!title || !content || !link || !image || !status || !expiration_date || !userId) {
      if (req.file) {
          require('fs').unlink(req.file.path, (err) => {
              if (err) console.error("Error deleting incomplete ad image:", err);
          });
      }
      return res.status(400).json({ error: "All required fields (title, content, link, image, status, expiration_date, user_id) are required" });
  }

  const createAdSql = `INSERT INTO ads (title, content, link, image, status, expiration_date, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)`; // เพิ่ม user_id
  pool.query(createAdSql, [title, content, link, image, status, expiration_date, userId], (err, results) => { // เพิ่ม userId
      if (err) {
          console.error("Database error during ad creation:", err);
          if (req.file) {
              require('fs').unlink(req.file.path, (unlinkErr) => {
                  if (unlinkErr) console.error("Error deleting ad image after DB error:", unlinkErr);
              });
          }
          return res.status(500).json({ error: "Error creating ad" });
      }

      res.status(201).json({ message: "Ad created successfully", ad_id: results.insertId });
  });
});


// สร้าง API สำหรับอัปเดตข้อมูล (Admin only)
app.put("/api/admin/ads/:id", authenticateToken, authorizeAdmin, upload.single('image'), (req, res) => {
  const { id } = req.params;
  const { title, content, link, status, expiration_date, admin_notes, show_at, expired_at } = req.body; // เพิ่ม expired_at
  const image = req.file ? `/uploads/${req.file.filename}` : null;

  // ถ้า status เป็น rejected แต่ไม่ได้ใส่ admin_notes ห้ามบันทึก
  if (status === 'rejected' && (!admin_notes || admin_notes.trim() === '')) {
    if (req.file) {
      require('fs').unlink(req.file.path, (err) => {
        if (err) console.error('Error deleting uploaded image:', err);
      });
    }
    return res.status(400).json({ error: 'กรุณาระบุเหตุผล (admin_notes) เมื่อปฏิเสธโฆษณา' });
  }

  const updateFields = [];
  const updateValues = [];
  if (title !== undefined) { updateFields.push('title = ?'); updateValues.push(title); }
  if (content !== undefined) { updateFields.push('content = ?'); updateValues.push(content); }
  if (link !== undefined) { updateFields.push('link = ?'); updateValues.push(link); }
  if (image) { updateFields.push('image = ?'); updateValues.push(image); }
  if (status !== undefined) { updateFields.push('status = ?'); updateValues.push(status); }
  if (expiration_date !== undefined) { updateFields.push('expiration_date = ?'); updateValues.push(expiration_date); }
  if (admin_notes !== undefined) { updateFields.push('admin_notes = ?'); updateValues.push(admin_notes); }
  if (show_at !== undefined) { updateFields.push('show_at = ?'); updateValues.push(show_at); }
  if (expired_at !== undefined) { updateFields.push('expired_at = ?'); updateValues.push(expired_at); }
  updateFields.push('updated_at = NOW()');

  if (updateFields.length === 1 && updateFields[0] === 'updated_at = NOW()') {
    if (req.file) {
      require('fs').unlink(req.file.path, (err) => {
        if (err) console.error('Error deleting uploaded image:', err);
      });
    }
    return res.status(400).json({ error: 'No meaningful fields to update besides updated_at' });
  }

  const sql = `UPDATE ads SET ${updateFields.join(', ')} WHERE id = ?`;
  updateValues.push(id);

  pool.query(sql, updateValues, (err, results) => {
    if (err) {
      console.error('Database error during ad update:', err);
      if (req.file) {
        require('fs').unlink(req.file.path, (unlinkErr) => {
          if (unlinkErr) console.error('Error deleting new ad image after DB update error:', unlinkErr);
        });
      }
      return res.status(500).json({ error: 'Error updating ad' });
    }
    if (results.affectedRows === 0) {
      if (req.file) {
        require('fs').unlink(req.file.path, (unlinkErr) => {
          if (unlinkErr) console.error('Error deleting new ad image for not found ad:', unlinkErr);
        });
      }
      return res.status(404).json({ error: 'Ad not found' });
    }
    // ถ้ามีการเปลี่ยน status ให้แจ้งเตือน user
    if (status !== undefined) {
      notifyAdsStatusChange(parseInt(id), status, admin_notes || null, (notifyErr) => {
        if (notifyErr) console.error('Notify user error:', notifyErr);
        return res.json({ message: 'Ad updated successfully and user notified.' });
      });
    } else {
      res.json({ message: 'Ad updated successfully' });
    }
  });
});


// Delete an Ad (Admin only)
app.delete("/api/ads/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;

  // ก่อนลบ ad ควรดึง path รูปภาพมาลบออกจาก server ด้วย
  const fetchImagePathSql = `SELECT image FROM ads WHERE id = ?`;
  pool.query(fetchImagePathSql, [id], (fetchErr, fetchResults) => {
      if (fetchErr) {
          console.error("Database error during fetching ad image for deletion:", fetchErr);
          return res.status(500).json({ error: "Error deleting ad (failed to fetch image path)" });
      }

      const imagePathToDelete = fetchResults.length > 0 ? fetchResults[0].image : null;

      const deleteAdSql = `DELETE FROM ads WHERE id = ?`;
      pool.query(deleteAdSql, [id], (err, results) => {
          if (err) {
              console.error("Database error during ad deletion:", err);
              return res.status(500).json({ error: "Error deleting ad" });
          }

          if (results.affectedRows === 0) {
              return res.status(404).json({ error: "Ad not found" });
          }

          // ลบไฟล์ภาพออกจาก server หลังจากลบข้อมูลใน DB สำเร็จ
          if (imagePathToDelete) {
              const fullPath = path.join(__dirname, imagePathToDelete); // Assuming imagePathToDelete is /uploads/filename.ext
              require('fs').unlink(fullPath, (unlinkErr) => {
                  if (unlinkErr) console.error("Error deleting ad image file from disk:", unlinkErr);
              });
          }

          res.json({ message: "Ad deleted successfully" });
      });
  });
});


// Get All Ads (Admin only) - ควรใช้ authenticateToken, authorizeAdmin
app.get("/api/ads", authenticateToken, authorizeAdmin, (req, res) => {
  const fetchAdsSql = `
      SELECT id, user_id, order_id, title, content, link, image, status, created_at, updated_at, expiration_date, admin_notes, show_at
      FROM ads
      ORDER BY
          FIELD(status, 'pending', 'paid', 'active', 'rejected'), -- Custom order for status
          created_at ASC; -- Oldest created_at first
  `;
  pool.query(fetchAdsSql, (err, results) => {
    if (err) {
      console.error("Database error during fetching ads:", err);
      return res.status(500).json({ error: "Error fetching ads" });
    }
    res.json(results);
  });
});


// Get Ad by ID (Admin only) - ควรใช้ authenticateToken, authorizeAdmin
app.get("/api/ads/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;

  const fetchAdSql = `SELECT * FROM ads WHERE id = ?`;
  pool.query(fetchAdSql, [id], (err, results) => {
      if (err) {
          console.error("Database error during fetching ad:", err);
          return res.status(500).json({ error: "Error fetching ad" });
      }

      if (results.length === 0) {
          return res.status(404).json({ error: "Ad not found" });
      }

      res.json(results[0]);
  });
});


// Serve Ad Image by ID (Admin only) - ควรใช้ authenticateToken, authorizeAdmin
app.get("/api/ads/:id/image", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;

  const fetchAdImageSql = `SELECT image FROM ads WHERE id = ?`;
  pool.query(fetchAdImageSql, [id], (err, results) => {
      if (err) {
          console.error("Database error during fetching ad image:", err);
          return res.status(500).json({ error: "Error fetching ad image" });
      }

      if (results.length === 0) {
          return res.status(404).json({ error: "Ad not found" });
      }

      const imagePath = results[0].image;
      if (imagePath) {
          res.json({ imageUrl: `${req.protocol}://${req.get('host')}${imagePath}` });
      } else {
          res.status(404).json({ error: "Image not found" });
      }
  });
});


// ดึงข้อมูลผู้ใช้ทั้งหมด (Admin only)
app.get("/api/admin/users", authenticateToken, authorizeAdmin, (req, res) => {
  const fetchUsersSql = "SELECT * FROM users";
  pool.query(fetchUsersSql, (err, results) => {
      if (err) {
          console.error("Database error during fetching users:", err);
          return res.status(500).json({ error: "Error fetching users" });
      }
      res.json(results);
  });
});


// ดึงข้อมูลผู้ใช้โดย ID (Admin only)
app.get("/api/admin/users/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;
  const fetchUserSql = "SELECT * FROM users WHERE id = ?";
  pool.query(fetchUserSql, [id], (err, results) => {
      if (err) {
          console.error("Database error during fetching user:", err);
          return res.status(500).json({ error: "Error fetching user" });
      }
      if (results.length === 0) {
          return res.status(404).json({ error: "User not found" });
      }
      res.json(results[0]);
  });
});


// Edit user status by admin (Admin only)
app.put("/api/admin/users/:id/status", authenticateToken, authorizeAdmin, (req, res) => {
   const { id } = req.params;
   const { status } = req.body;
  
   if (!status) {
       return res.status(400).json({ error: "Status is required" });
   }
   // Removed updated_at from the SQL query
   const updateStatusSql = "UPDATE users SET status = ? WHERE id = ?";
   pool.query(updateStatusSql, [status, id], (err, results) => {
       if (err) {
           console.error("Database error during user status update:", err);
           return res.status(500).json({ error: "Error updating user status" });
       }
       if (results.affectedRows === 0) {
           return res.status(404).json({ error: "User not found" });
       }
       res.json({ message: "User status updated successfully" });
   });
});


// Soft Delete a User, Hard Delete their Posts, and Delete Follows (Admin-Only)
app.delete("/api/admin/users/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;

  pool.getConnection((err, connection) => {
    if (err) {
      console.error('Error getting DB connection:', err);
      return res.status(500).json({ error: 'Database connection error.' });
    }

    connection.beginTransaction((err) => {
      if (err) {
        connection.release();
        console.error('Error starting transaction:', err);
        return res.status(500).json({ error: 'Failed to start transaction.' });
      }

      // Delete all posts of the user (hard delete)
      const deletePostsSql = "DELETE FROM posts WHERE user_id = ?";
      connection.query(deletePostsSql, [id], (err, postResults) => {
        if (err) {
          return connection.rollback(() => {
            connection.release();
            console.error("Error deleting posts:", err);
            res.status(500).json({ error: "Failed to delete user's posts." });
          });
        }

        // Delete all follows of the user (both following and followers)
        const deleteFollowsSql = "DELETE FROM follower_following WHERE follower_id = ? OR following_id = ?";
        connection.query(deleteFollowsSql, [id, id], (err, followResults) => {
          if (err) {
            return connection.rollback(() => {
              connection.release();
              console.error("Error deleting follows:", err);
              res.status(500).json({ error: "Failed to delete user's follows." });
            });
          }

          // Soft delete the user (update status to 'deactivated')
          // ไม่มี updated_at ในคำสั่ง SQL นี้แล้ว
          const softDeleteUserSql = "UPDATE users SET status = 'deactivated' WHERE id = ?";
          connection.query(softDeleteUserSql, [id], (err, userResults) => {
            if (err) {
              return connection.rollback(() => {
                connection.release();
                console.error("Error soft-deleting user:", err);
                res.status(500).json({ error: "Failed to soft-delete user." });
              });
            }

            if (userResults.affectedRows === 0) {
              return connection.rollback(() => {
                connection.release();
                res.status(404).json({ error: "User not found" });
              });
            }

            connection.commit((err) => {
              if (err) {
                return connection.rollback(() => {
                  connection.release();
                  console.error("Error committing transaction:", err);
                  res.status(500).json({ error: "Transaction failed." });
                });
              }
              connection.release();
              res.json({
                message: "User soft-deleted, their posts and follows deleted successfully",
                deletedPostsCount: postResults.affectedRows,
                deletedFollowsCount: followResults.affectedRows
              });
            });
          });
        });
      });
    });
  });
});


// Get all posts (Admin only)
app.get("/api/admin/posts", authenticateToken, authorizeAdmin, (req, res) => {
  const fetchPostsSql = "SELECT * FROM posts ORDER BY created_at DESC"; // เพิ่ม ORDER BY
  pool.query(fetchPostsSql, (err, results) => {
      if (err) {
          console.error("Database error during fetching posts:", err);
          return res.status(500).json({ error: "Error fetching posts" });
      }
      res.json(results);
  });
});


// Get post by ID (Admin only)
app.get("/api/admin/posts/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;
  const fetchPostSql = "SELECT * FROM posts WHERE id = ?";
  pool.query(fetchPostSql, [id], (err, results) => {
      if (err) {
          console.error("Database error during fetching post:", err);
          return res.status(500).json({ error: "Error fetching post" });
      }
      if (results.length === 0) {
          return res.status(404).json({ error: "Post not found" });
      }
      res.json(results[0]);
  });
});


// Update post status by admin (Admin only)
app.put("/api/admin/posts/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;
  const { Title, content, status, ProductName } = req.body;

  // Build dynamic update query
  const updateFields = [];
  const updateValues = [];

  if (Title !== undefined) { updateFields.push('Title = ?'); updateValues.push(Title); }
  if (content !== undefined) { updateFields.push('content = ?'); updateValues.push(content); }
  if (status !== undefined) { updateFields.push('status = ?'); updateValues.push(status); }
  if (ProductName !== undefined) { updateFields.push('ProductName = ?'); updateValues.push(ProductName); }

  updateFields.push('updated_at = NOW()'); // Always update updated_at

  if (updateFields.length === 1 && updateFields[0] === 'updated_at = NOW()') {
      return res.status(400).json({ error: 'No meaningful fields to update besides updated_at' });
  }

  const updatePostSql = `
      UPDATE posts 
      SET ${updateFields.join(', ')} 
      WHERE id = ?`;
  updateValues.push(id);

  pool.query(updatePostSql, updateValues, (err, results) => {
      if (err) {
          console.error("Database error during updating post:", err);
          return res.status(500).json({ error: "Error updating post" });
      }
      if (results.affectedRows === 0) {
          return res.status(404).json({ error: "Post not found" });
      }
      res.json({ message: "Post updated successfully" });
  });
});


// Delete post by admin (Admin only)
app.delete("/api/admin/posts/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;

  // เริ่มต้น Transaction (Optional แต่แนะนำสำหรับ Multiple Operations)
  pool.getConnection((err, connection) => {
      if (err) {
          console.error("Error getting database connection:", err);
          return res.status(500).json({ error: "Database connection error" });
      }

      connection.beginTransaction(err => {
          if (err) {
              connection.release();
              console.error("Error starting transaction:", err);
              return res.status(500).json({ error: "Failed to start transaction" });
          }

          // 1. ลบรายงานทั้งหมดที่เกี่ยวข้องกับ post_id นี้ก่อน
          const deleteReportsSql = "DELETE FROM reports WHERE post_id = ?";
          connection.query(deleteReportsSql, [id], (err, reportsResults) => {
              if (err) {
                  return connection.rollback(() => {
                      connection.release();
                      console.error("Database error during reports deletion:", err);
                      res.status(500).json({ error: "Error deleting related reports" });
                  });
              }

              // 2. เมื่อลบรายงานแล้ว ค่อยลบโพสต์
              const deletePostSql = "DELETE FROM posts WHERE id = ?";
              connection.query(deletePostSql, [id], (err, postResults) => {
                  if (err) {
                      return connection.rollback(() => {
                          connection.release();
                          console.error("Database error during post deletion:", err);
                          res.status(500).json({ error: "Error deleting post" });
                      });
                  }

                  if (postResults.affectedRows === 0) {
                      return connection.rollback(() => {
                          connection.release();
                          res.status(404).json({ error: "Post not found" });
                      });
                  }

                  // Commit Transaction หากทุกอย่างสำเร็จ
                  connection.commit(err => {
                      if (err) {
                          return connection.rollback(() => {
                              connection.release();
                              console.error("Error committing transaction:", err);
                              res.status(500).json({ error: "Failed to commit transaction" });
                          });
                      }
                      connection.release();
                      res.json({ message: "Post and related reports deleted successfully" });
                  });
              });
          });
      });
  });
});


// Get all reported posts (Admin only)
app.get("/api/admin/reported-posts", authenticateToken, authorizeAdmin, (req, res) => {
  pool.getConnection((err, connection) => {
      if (err) {
          console.error("Error getting database connection:", err);
          return res.status(500).json({ error: "Failed to connect to database." });
      }

      const sql = `
          SELECT
              r.id AS report_id,
              r.post_id,
              r.user_id,
              r.reason,
              r.reported_at,
              r.status,
              p.id AS actual_post_id,
              p.title AS post_title,
              p.content AS post_content,
              p.photo_url AS post_image_url,
              u.username AS reported_by_username
          FROM
              reports r
          JOIN
              posts p ON r.post_id = p.id
          JOIN
              users u ON r.user_id = u.id
          ORDER BY
              CASE r.status
                  WHEN 'pending' THEN 1
                  WHEN 'normally' THEN 2
                  WHEN 'block' THEN 3
                  ELSE 99
              END,
              r.reported_at DESC;
      `;

      connection.query(sql, (error, results) => {
          connection.release(); // Always release the connection
          if (error) {
              console.error("Error fetching reported posts:", error);
              return res.status(500).json({ error: "Error fetching reported posts: " + error.message });
          }
          console.log("Fetched Reported Posts:", results);
          res.json(results);
      });
  });
});


// Update report status (Admin only)
app.put("/api/admin/reports/:reportId", authenticateToken, authorizeAdmin, (req, res) => {
  const { reportId } = req.params;
  const { status: newStatus } = req.body; // newStatus can be 'pending', 'block', 'normally'

  pool.getConnection((err, connection) => {
      if (err) {
          console.error("Error getting database connection:", err);
          return res.status(500).json({ error: "Failed to connect to database." });
      }

      connection.beginTransaction((err) => {
          if (err) {
              connection.release();
              console.error("Error starting transaction:", err);
              return res.status(500).json({ error: "Failed to start database transaction." });
          }

          // 1. Get the post_id associated with this specific reportId
          connection.query('SELECT post_id FROM reports WHERE id = ?', [reportId], (error, reportRows) => {
              if (error) {
                  return connection.rollback(() => {
                      connection.release();
                      console.error("Error fetching report for update:", error);
                      res.status(500).json({ error: 'Failed to fetch report details: ' + error.message });
                  });
              }
              if (reportRows.length === 0) {
                  return connection.rollback(() => {
                      connection.release();
                      res.status(404).json({ error: 'Report not found.' });
                  });
              }
              const postId = reportRows[0].post_id;

              // 2. Conditional Update Logic based on newStatus
              if (newStatus === 'block') {
                  // Update ALL reports for this post_id to 'block'
                  connection.query('UPDATE reports SET status = ? WHERE post_id = ?', ['block', postId], (errReports) => {
                      if (errReports) {
                          return connection.rollback(() => {
                              connection.release();
                              console.error('Error updating reports to block:', errReports);
                              res.status(500).json({ error: 'Failed to update reports: ' + errReports.message });
                          });
                      }
                      // Update the post status in 'posts' table to 'deactivate'
                      connection.query('UPDATE posts SET status = ? WHERE id = ?', ['deactivate', postId], (errPosts) => {
                          if (errPosts) {
                              return connection.rollback(() => {
                                  connection.release();
                                  console.error('Error deactivating post:', errPosts);
                                  res.status(500).json({ error: 'Failed to deactivate post: ' + errPosts.message });
                              });
                          }
                          connection.commit((errCommit) => {
                              if (errCommit) {
                                  return connection.rollback(() => {
                                      connection.release();
                                      console.error('Error committing transaction (block):', errCommit);
                                      res.status(500).json({ error: 'Transaction failed during commit: ' + errCommit.message });
                                  });
                              }
                              connection.release();
                              res.json({ message: 'Report status and related post status updated successfully to block/deactivate.' });
                          });
                      });
                  });
              } else if (newStatus === 'normally') {
                  // Update ALL reports for this post_id to 'normally'
                  connection.query('UPDATE reports SET status = ? WHERE post_id = ?', ['normally', postId], (errReports) => {
                      if (errReports) {
                          return connection.rollback(() => {
                              connection.release();
                              console.error('Error updating reports to normally:', errReports);
                              res.status(500).json({ error: 'Failed to update reports: ' + errReports.message });
                          });
                      }
                      // Update the post status in 'posts' table to 'active'
                      connection.query('UPDATE posts SET status = ? WHERE id = ?', ['active', postId], (errPosts) => {
                          if (errPosts) {
                              return connection.rollback(() => {
                                  connection.release();
                                  console.error('Error activating post:', errPosts);
                                  res.status(500).json({ error: 'Failed to activate post: ' + errPosts.message });
                              });
                          }
                          connection.commit((errCommit) => {
                              if (errCommit) {
                                  return connection.rollback(() => {
                                      connection.release();
                                      console.error('Error committing transaction (normally):', errCommit);
                                      res.status(500).json({ error: 'Transaction failed during commit: ' + errCommit.message });
                                  });
                              }
                              connection.release();
                              res.json({ message: 'Report status and related post status updated successfully to normally/active.' });
                          });
                      });
                  });
              } else if (newStatus === 'pending') {
                  // <<< CORRECTED: Update ALL reports for this post_id to 'pending'
                  connection.query('UPDATE reports SET status = ? WHERE post_id = ?', ['pending', postId], (errReports) => {
                      if (errReports) {
                          return connection.rollback(() => {
                              connection.release();
                              console.error('Error updating reports to pending:', errReports);
                              res.status(500).json({ error: 'Failed to update reports to pending: ' + errReports.message });
                          });
                      }
                      connection.commit((errCommit) => {
                          if (errCommit) {
                              return connection.rollback(() => {
                                  connection.release();
                                  console.error('Error committing transaction (pending):', errCommit);
                                  res.status(500).json({ error: 'Transaction failed during commit: ' + errCommit.message });
                              });
                          }
                          connection.release();
                          res.json({ message: 'All related report statuses updated to pending.' });
                      });
                  });
              } else {
                  return connection.rollback(() => {
                      connection.release();
                      res.status(400).json({ error: 'Invalid status provided.' });
                  });
              }
          });
      });
  });
});


// Get All Categories (Admin only)
app.get("/api/categories", authenticateToken, authorizeAdmin, (req, res) => {
  const fetchCategoriesSql = 'SELECT * FROM category ORDER BY CategoryID ASC';
  pool.query(fetchCategoriesSql, (err, results) => {
      if (err) {
          console.error("Database error during fetching categories:", err);
          return res.status(500).json({ error: "Error fetching categories" });
      }
      res.json(results);
  });
});


// Create a Category (Admin only)
app.post("/api/categories", authenticateToken, authorizeAdmin, (req, res) => {
  const { CategoryName } = req.body;

  if (!CategoryName) {
      return res.status(400).json({ error: "CategoryName is required" });
  }

  const createCategorySql = 'INSERT INTO category (CategoryName, created_at, updated_at) VALUES (?, NOW(), NOW())'; // เพิ่ม created_at, updated_at
  pool.query(createCategorySql, [CategoryName], (err, results) => {
      if (err) {
          console.error("Database error during category creation:", err);
          return res.status(500).json({ error: "Error creating category" });
      }
      res.status(201).json({ message: "Category created successfully", categoryId: results.insertId });
  });
});


// Update a Category (Admin only)
app.put("/api/categories/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;
  const { CategoryName } = req.body;

  if (!CategoryName) {
      return res.status(400).json({ error: "CategoryName is required" });
  }

  const updateCategorySql = 'UPDATE category SET CategoryName = ?, updated_at = NOW() WHERE CategoryID = ?'; // เพิ่ม updated_at
  pool.query(updateCategorySql, [CategoryName, id], (err, results) => {
      if (err) {
          console.error("Database error during category update:", err);
          return res.status(500).json({ error: "Error updating category" });
      }
      if (results.affectedRows === 0) {
          return res.status(404).json({ error: "Category not found" });
      }
      res.json({ message: "Category updated successfully" });
  });
});


// Delete a Category (Admin only)
app.delete("/api/categories/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;

  const deleteCategorySql = 'DELETE FROM category WHERE CategoryID = ?';
  pool.query(deleteCategorySql, [id], (err, results) => {
      if (err) {
          console.error("Database error during category deletion:", err);
          return res.status(500).json({ error: "Error deleting category" });
      }
      if (results.affectedRows === 0) {
          return res.status(404).json({ error: "Category not found" });
      }
      res.json({ message: "Category deleted successfully" });
  });
});


// API สำหรับแอดมินในการอัปเดตสถานะโพสต์เป็น 'deactivate' และลบรายงานที่เกี่ยวข้อง
app.put("/api/admin/update/poststatus", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.body;

  if (!id) {
      return res.status(400).json({ error: "Post ID is required" });
  }

  pool.getConnection((err, connection) => {
      if (err) {
          console.error('Error getting DB connection:', err);
          return res.status(500).json({ error: 'Database connection error.' });
      }

      connection.beginTransaction(async (err) => {
          if (err) {
              connection.release();
              console.error('Error starting transaction:', err);
              return res.status(500).json({ error: 'Failed to start transaction.' });
          }

          try {
              // Check if the post exists in the 'reports' table and its status (optional, but good for logic)
              const checkPostInReportsSql = "SELECT * FROM reports WHERE post_id = ?";
              const [checkResults] = await connection.execute(checkPostInReportsSql, [id]);

              if (checkResults.length === 0) {
                  await connection.rollback();
                  connection.release();
                  return res.status(404).json({ error: "Post not found in pending reports (or already handled)" });
              }

              // Update post status to 'deactivate'
              const updatePostStatusSql = `
                  UPDATE posts 
                  SET status = 'deactivate', updated_at = NOW()
                  WHERE id = ?;
              `;
              const [postUpdateResults] = await connection.execute(updatePostStatusSql, [id]);

              if (postUpdateResults.affectedRows === 0) {
                  await connection.rollback();
                  connection.release();
                  return res.status(404).json({ error: "Post not found or already deactivated" });
              }

              // Delete the related report(s)
              const deleteReportSql = "DELETE FROM reports WHERE post_id = ?";
              const [reportDeleteResults] = await connection.execute(deleteReportSql, [id]);

              await connection.commit();
              connection.release();
              res.json({ message: "Post status updated to deactivate successfully and associated reports deleted", deletedReportsCount: reportDeleteResults.affectedRows });

          } catch (transactionErr) {
              await connection.rollback();
              connection.release();
              console.error("Transaction failed during post status update and report deletion:", transactionErr);
              res.status(500).json({ error: "Transaction failed during post status update and report deletion" });
          }
      });
  });
});


// APi สำหรับแอดมินดูข้อมูลออเดอร์ทั้งหมด (Admin only)
app.get("/api/admin/orders", authenticateToken, authorizeAdmin, (req, res) => {
  const sql = `
      SELECT o.*, a.title, a.content, a.link, a.image, a.status AS ad_status
      FROM orders o
      LEFT JOIN ads a ON o.id = a.order_id
      ORDER BY o.created_at DESC
  `;
  pool.query(sql, (err, results) => {
      if (err) {
          console.error(`[ERROR] Database error fetching all orders:`, err);
          return res.status(500).json({ error: 'Database error' });
      }
      res.json(results);
  });
});


// PUT /api/admin/orders/:orderId - แก้ไขข้อมูลออเดอร์ (Admin only)
app.put("/api/admin/orders/:orderId", authenticateToken, authorizeAdmin, async (req, res) => {
  const { orderId } = req.params;
  const { amount, status, prompay_number, title, content, link, image, expiration_date, admin_notes } = req.body;

  // 1. ถ้า status เป็น rejected แต่ไม่ได้ใส่ admin_notes ห้ามบันทึก
  if (status === 'rejected' && (!admin_notes || admin_notes.trim() === '')) {
    return res.status(400).json({ error: 'กรุณาระบุเหตุผล (admin_notes) เมื่อปฏิเสธออเดอร์' });
  }

  pool.getConnection((err, connection) => {
    if (err) {
      console.error('Error getting DB connection:', err);
      return res.status(500).json({ error: 'Database connection error.' });
    }
    connection.beginTransaction(err => {
      if (err) {
        connection.release();
        return res.status(500).json({ error: 'Database transaction error.' });
      }
      // 1. อัปเดต orders
      const updateOrderSql = `UPDATE orders SET amount = COALESCE(?, amount), status = COALESCE(?, status), prompay_number = COALESCE(?, prompay_number), updated_at = NOW() WHERE id = ?`;
      connection.query(updateOrderSql, [amount, status, prompay_number, orderId], (err, orderResult) => {
        if (err) {
          return connection.rollback(() => {
            connection.release();
            res.status(500).json({ error: 'Failed to update order.' });
          });
        }
        // 2. อัปเดต ads ที่เชื่อมโยง (ถ้ามีข้อมูลส่งมา)
        if (title !== undefined || content !== undefined || link !== undefined || image !== undefined || status !== undefined || expiration_date !== undefined || admin_notes !== undefined) {
          const updateFields = [];
          const updateValues = [];
          if (title !== undefined) { updateFields.push('title = ?'); updateValues.push(title); }
          if (content !== undefined) { updateFields.push('content = ?'); updateValues.push(content); }
          if (link !== undefined) { updateFields.push('link = ?'); updateValues.push(link); }
          if (image !== undefined) { updateFields.push('image = ?'); updateValues.push(image); }
          if (status !== undefined) { updateFields.push('status = ?'); updateValues.push(status); }
          if (expiration_date !== undefined) { updateFields.push('expiration_date = ?'); updateValues.push(expiration_date); }
          if (admin_notes !== undefined) { updateFields.push('admin_notes = ?'); updateValues.push(admin_notes); }
          if (updateFields.length > 0) {
            updateFields.push('updated_at = NOW()');
            const updateAdSql = `UPDATE ads SET ${updateFields.join(', ')} WHERE order_id = ?`;
            updateValues.push(orderId);
            connection.query(updateAdSql, updateValues, (err, adResult) => {
              if (err) {
                return connection.rollback(() => {
                  connection.release();
                  res.status(500).json({ error: 'Failed to update ad.' });
                });
              }
              // --- แจ้งเตือน user ด้วยฟังก์ชันกลาง ---
              const getAdIdSql = 'SELECT id FROM ads WHERE order_id = ? LIMIT 1';
              connection.query(getAdIdSql, [orderId], (adIdErr, adIdRows) => {
                if (!adIdErr && adIdRows.length > 0 && status) {
                  const adId = adIdRows[0].id;
                  notifyAdsStatusChange(adId, status, admin_notes || null, (notifyErr) => {
                    if (notifyErr) console.error('Notify user error:', notifyErr);
                  });
                }
                connection.commit(commitErr => {
                  if (commitErr) {
                    return connection.rollback(() => {
                      connection.release();
                      res.status(500).json({ error: 'Transaction commit failed.' });
                    });
                  }
                  connection.release();
                  res.json({ message: 'Order and related ad updated successfully.' });
                });
              });
            });
            return;
          }
        }
        // ถ้าไม่มีข้อมูล ads ให้ commit เลย
        connection.commit(commitErr => {
          if (commitErr) {
            return connection.rollback(() => {
              connection.release();
              res.status(500).json({ error: 'Transaction commit failed.' });
            });
          }
          connection.release();
          res.json({ message: 'Order updated successfully.' });
        });
      });
    });
  });
});

// DELETE /api/admin/orders/:orderId - ลบออเดอร์และโฆษณาที่เกี่ยวข้อง (Admin only)
app.delete("/api/admin/orders/:orderId", authenticateToken, authorizeAdmin, (req, res) => {
  const { orderId } = req.params;
  pool.getConnection((err, connection) => {
    if (err) {
      console.error('Error getting DB connection:', err);
      return res.status(500).json({ error: 'Database connection error.' });
    }
    connection.beginTransaction(err => {
      if (err) {
        connection.release();
        return res.status(500).json({ error: 'Database transaction error.' });
      }
      // 1. ลบ ads ที่เชื่อมโยงกับ order นี้ก่อน
      const fetchAdSql = 'SELECT image FROM ads WHERE order_id = ?';
      connection.query(fetchAdSql, [orderId], (fetchErr, adResults) => {
        if (fetchErr) {
          return connection.rollback(() => {
            connection.release();
            res.status(500).json({ error: 'Failed to fetch ad for deletion.' });
          });
        }
        // เตรียมลบไฟล์ภาพ ads ถ้ามี
        const imagePaths = adResults.map(row => row.image).filter(Boolean);
        const deleteAdSql = 'DELETE FROM ads WHERE order_id = ?';
        connection.query(deleteAdSql, [orderId], (adDelErr, adDelResult) => {
          if (adDelErr) {
            return connection.rollback(() => {
              connection.release();
              res.status(500).json({ error: 'Failed to delete ad.' });
            });
          }
          // 2. ลบ order
          const deleteOrderSql = 'DELETE FROM orders WHERE id = ?';
          connection.query(deleteOrderSql, [orderId], (orderDelErr, orderDelResult) => {
            if (orderDelErr) {
              return connection.rollback(() => {
                connection.release();
                res.status(500).json({ error: 'Failed to delete order.' });
              });
            }
            connection.commit(commitErr => {
              if (commitErr) {
                return connection.rollback(() => {
                  connection.release();
                  res.status(500).json({ error: 'Transaction commit failed.' });
                });
              }
              connection.release();
              // ลบไฟล์ภาพ ads ออกจาก disk
              const fs = require('fs');
              const path = require('path');
              imagePaths.forEach(imgPath => {
                if (imgPath) {
                  const fullPath = path.join(__dirname, imgPath);
                  fs.unlink(fullPath, (err) => {
                    if (err) console.error('Error deleting ad image file:', err);
                  });
                }
              });
              res.json({ message: 'Order and related ad(s) deleted successfully.' });
            });
          });
        });
      });
    });
  });
});


//########################################################   Admin Search API  ########################################################


//search ads
app.get("/api/admin/search/ads", authenticateToken, (req, res) => {
  const { q: query } = req.query;

  if (!query) {
    return res.status(400).json({ error: 'Search query is required' });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`;

  const searchSql = `
    SELECT
      id,
      title,
      content,
      link,
      status,
      expiration_date,
      image,
      created_at,
      updated_at
    FROM ads
    WHERE (
      LOWER(title) LIKE ? OR
      LOWER(content) LIKE ? OR
      LOWER(link) LIKE ? OR
      LOWER(status) LIKE ? OR
      id LIKE ?
    )
    ORDER BY created_at DESC;
  `;

  pool.query(
    searchSql,
    [searchValue, searchValue, searchValue, searchValue, searchValue],
    (err, results) => {
      if (err) {
        console.error("Database error during ads search:", err);
        return res.status(500).json({ error: "Internal server error" });
      }

      if (results.length === 0) {
        return res.status(200).json({ message: "No advertisements found", results: [] });
      }

      // ส่ง image ตรงๆ ไม่เติม path
      const adsWithImagePaths = results.map(ad => ({
        ...ad,
        image: ad.image ? ad.image : null
      }));

      res.status(200).json(adsWithImagePaths);
    }
  );
});


//search users
app.get("/api/admin/search/users", authenticateToken, (req, res) => {
  const { q: query } = req.query;

  if (!query) {
    return res.status(400).json({ error: 'Search query is required' });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`;

  const searchSql = `
    SELECT
      id,
      email,
      username,
      picture,
      created_at,
      last_login,
      last_login_ip,
      gender,
      bio,
      status,
      role,
      birthday,
      age
    FROM users
    WHERE (
      LOWER(email) LIKE ? OR
      LOWER(username) LIKE ? OR
      LOWER(gender) LIKE ? OR
      LOWER(bio) LIKE ? OR
      LOWER(status) LIKE ? OR
      LOWER(role) LIKE ? OR
      id LIKE ?
    )
    ORDER BY created_at DESC;
  `;

  const queryParams = [
    searchValue, searchValue, searchValue, searchValue,
    searchValue, searchValue, searchValue
  ];

  pool.query(
    searchSql,
    queryParams,
    (err, results) => {
      if (err) {
        console.error("Database error during users search:", err);
        return res.status(500).json({ error: "Internal server error" });
      }

      // map picture ให้เป็น path เสมอ (กันกรณี picture เป็น string ว่าง/null)
      const usersWithImagePaths = results.map(user => ({
        ...user,
        picture: user.picture && user.picture.trim() !== ''
          ? `${user.picture}`
          : null
      }));

      // log ข้อมูลที่ส่งกลับ
      console.log("search results:", usersWithImagePaths);

      if (usersWithImagePaths.length === 0) {
        return res.status(200).json({ message: "No users found", results: [] });
      }

      res.status(200).json(usersWithImagePaths);
    }
  );
});


//search posts
app.get("/api/admin/search/posts", authenticateToken, (req, res) => {
  const { q: query } = req.query;

  if (!query) {
    return res.status(400).json({ error: 'Search query is required' });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`;

  const searchSql = `
    SELECT
      id,
      user_id,
      content,
      video_url,
      photo_url,
      Title,
      CategoryID,
      ProductName,
      created_at,
      updated_at,
      status
    FROM posts
    WHERE (
      LOWER(content) LIKE ? OR
      LOWER(Title) LIKE ? OR
      LOWER(ProductName) LIKE ? OR
      LOWER(status) LIKE ? OR
      id LIKE ? OR
      user_id LIKE ?
    )
    ORDER BY created_at DESC;
  `;

  const queryParams = [
    searchValue, // content
    searchValue, // Title
    searchValue, // ProductName
    searchValue, // status
    searchValue, // id
    searchValue  // user_id
  ];

  pool.query(
    searchSql,
    queryParams,
    (err, results) => {
      if (err) {
        console.error("Database error during posts search:", err);
        return res.status(500).json({ error: "Internal server error" });
      }

      if (results.length === 0) {
        return res.status(200).json({ message: "No posts found", results: [] });
      }

      // ส่งข้อมูลตรงๆ ไม่เติม path ใดๆ
      res.status(200).json(results);
    }
  );
});


//search reports
app.get("/api/admin/search/reports", authenticateToken, (req, res) => {
  const { q: query } = req.query;

  if (!query) {
    return res.status(400).json({ error: 'Search query is required' });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`;

  // JOIN post และ user เพื่อดึงข้อมูลที่ UI ต้องการ
  const searchSql = `
    SELECT
      r.id AS report_id,
      r.user_id AS reported_by_user_id,
      u.username AS reported_by_username,
      r.post_id AS actual_post_id,
      p.Title AS post_title,
      p.photo_url AS post_image_url,
      r.reason,
      r.reported_at,
      r.status
    FROM reports r
    LEFT JOIN users u ON r.user_id = u.id
    LEFT JOIN posts p ON r.post_id = p.id
    WHERE (
      LOWER(r.reason) LIKE ? OR
      LOWER(r.status) LIKE ? OR
      r.id LIKE ? OR
      r.user_id LIKE ? OR
      r.post_id LIKE ? OR
      LOWER(u.username) LIKE ? OR
      LOWER(p.Title) LIKE ?
    )
    ORDER BY r.reported_at DESC;
  `;

  const queryParams = [
    searchValue, // reason
    searchValue, // status
    searchValue, // report id
    searchValue, // user id
    searchValue, // post id
    searchValue, // username
    searchValue  // post title
  ];

  pool.query(
    searchSql,
    queryParams,
    (err, results) => {
      if (err) {
        console.error("Database error during reports search:", err);
        return res.status(500).json({ error: "Internal server error" });
      }

      // แปลง photo_url จาก string เป็น array ถ้าเก็บเป็น JSON string
      const mapped = results.map(row => ({
        ...row,
        post_image_url: row.post_image_url
          ? (typeof row.post_image_url === 'string'
              ? JSON.parse(row.post_image_url)
              : row.post_image_url)
          : undefined
      }));

      if (mapped.length === 0) {
        return res.status(200).json({ message: "No reports found", results: [] });
      }

      res.status(200).json(mapped);
    }
  );
});


//########################################################   Message  API  ########################################################


// API สร้าง Match อัตโนมัติเมื่อมีการ Follow
app.post("/api/users/:userId/follow/:followingId", (req, res) => {
    const { userId, followingId } = req.params;

    // เพิ่มข้อมูลการ follow ใน table follower_following
    const followQuery = `
        INSERT INTO follower_following (follower_id, following_id, follow_date)
        VALUES (?, ?, NOW())
        ON DUPLICATE KEY UPDATE follow_date = follow_date
    `;

    pool.query(followQuery, [userId, followingId], (err, result) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).json({ error: 'Database error' });
        }

        // สร้าง match สำหรับ chat อัตโนมัติ
        const createMatchQuery = `
            INSERT INTO matches (user1ID, user2ID, matchDate)
            VALUES (?, ?, NOW())
            ON DUPLICATE KEY UPDATE matchDate = matchDate
        `;

        pool.query(createMatchQuery, [userId, followingId], (err, matchResult) => {
            if (err) {
                console.error('Error creating match:', err);
                // ไม่ return error เพราะการ follow สำเร็จแล้ว
            }
            
            res.status(200).json({ 
                message: 'Followed successfully',
                matchID: matchResult ? matchResult.insertId : null
            });
        });
    });
});


// API สร้าง Match จากการ Follow (เรียกแยกได้ถ้าต้องการ)
app.post("/api/create-match-on-follow", (req, res) => {
    const { followerID, followingID } = req.body;

    if (!followerID || !followingID) {
        return res.status(400).json({ error: 'Missing followerID or followingID' });
    }

    // ตรวจสอบว่ามีการ follow กันหรือไม่
    const checkFollowQuery = `
        SELECT * FROM follower_following 
        WHERE follower_id = ? AND following_id = ?
    `;

    pool.query(checkFollowQuery, [followerID, followingID], (err, followResults) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).json({ error: 'Database error' });
        }

        if (followResults.length === 0) {
            return res.status(403).json({ error: 'User must follow first before creating chat' });
        }

        // ตรวจสอบว่ามี match อยู่แล้วหรือไม่
        const checkMatchQuery = `
            SELECT matchID FROM matches 
            WHERE (user1ID = ? AND user2ID = ?) 
               OR (user1ID = ? AND user2ID = ?)
        `;

        pool.query(checkMatchQuery, [followerID, followingID, followingID, followerID], (err, results) => {
            if (err) {
                console.error('Database error:', err);
                return res.status(500).json({ error: 'Database error' });
            }

            if (results.length > 0) {
                return res.status(200).json({ 
                    success: 'Match already exists', 
                    matchID: results[0].matchID 
                });
            }

            // สร้าง match ใหม่
            const createMatchQuery = `
                INSERT INTO matches (user1ID, user2ID, matchDate)
                VALUES (?, ?, NOW())
            `;

            pool.query(createMatchQuery, [followerID, followingID], (err, result) => {
                if (err) {
                    console.error('Database error:', err);
                    return res.status(500).json({ error: 'Database error' });
                }

                res.status(201).json({ 
                    success: 'Match created successfully', 
                    matchID: result.insertId 
                });
            });
        });
    });
});


// API Get Matches - แสดงรายการ chat ของ user
app.get("/api/matches/:userID", (req, res) => {
    const { userID } = req.params;

    const getMatchedUsersWithLastMessageQuery = `
        SELECT 
            u.id AS userID,
            u.username AS nickname,
            u.picture AS imageFile,
            (SELECT c.message FROM chats c WHERE c.matchID = m.matchID ORDER BY c.timestamp DESC LIMIT 1) AS lastMessage,
            m.matchID,
            DATE_FORMAT(GREATEST(
                COALESCE((SELECT c.timestamp FROM chats c WHERE c.matchID = m.matchID ORDER BY c.timestamp DESC LIMIT 1), '1970-01-01 00:00:00'), 
                m.matchDate), '%H:%i') AS lastInteraction,
            GREATEST(
                COALESCE((SELECT c.timestamp FROM chats c WHERE c.matchID = m.matchID ORDER BY c.timestamp DESC LIMIT 1), '1970-01-01 00:00:00'), 
                m.matchDate) AS fullLastInteraction,
            COALESCE(b.isBlocked, 0) AS isBlocked,
            CASE 
                WHEN ff.follower_id IS NOT NULL THEN 1
                ELSE 0
            END AS isFollowing
        FROM matches m
        JOIN users u ON (m.user1ID = u.id OR m.user2ID = u.id)
        LEFT JOIN deleted_chats d ON d.matchID = m.matchID AND d.userID = ?
        LEFT JOIN blocked_chats b ON b.matchID = m.matchID AND (b.user1ID = ? OR b.user2ID = ?)
        LEFT JOIN follower_following ff ON ff.follower_id = ? AND ff.following_id = u.id
        WHERE (m.user1ID = ? OR m.user2ID = ?)
          AND u.id != ?
          AND (d.deleted IS NULL OR d.deleted = 0 OR (SELECT COUNT(*) FROM chats c WHERE c.matchID = m.matchID AND c.timestamp > d.deleteTimestamp) > 0) 
        ORDER BY fullLastInteraction DESC;
    `;

    pool.query(getMatchedUsersWithLastMessageQuery, [userID, userID, userID, userID, userID, userID, userID], (err, results) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).json({ error: 'Database error' });
        }

        results.forEach(user => {
            if (user.imageFile && !user.imageFile.startsWith('http')) {
                user.imageFile = `${req.protocol}://${req.get('host')}${user.imageFile}`;
            }

            if (user.lastMessage === null) {
                user.lastMessage = "Start chatting!";
            }
        });

        return res.status(200).json(results);
    });
});


//get chats
app.get("/api/chats/:matchID", (req, res) => {
    const { matchID } = req.params;

    const getChatQuery = `
        SELECT 
            c.senderID, 
            u.username AS nickname,
            u.picture AS imageFile,
            c.message, 
            c.timestamp 
        FROM chats c
        JOIN users u ON c.senderID = u.id
        WHERE c.matchID = ?
        ORDER BY c.timestamp ASC;
    `;

    pool.query(getChatQuery, [matchID], (err, results) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).json({ error: 'Database error' });
        }

        results.forEach(chat => {
            if (chat.imageFile) {
                chat.imageFile = `${req.protocol}://${req.get('host')}${chat.imageFile}`;
            }
        });

        // Backend ส่ง JSON ในรูปแบบนี้:
        return res.status(200).json({ messages: results }); // <--- ตรงนี้
    });
});


// API Send Chat Message
app.post("/api/chats/:matchID", (req, res) => {
    const { matchID } = req.params;
    const { senderID, message } = req.body;

    if (!senderID || !message) {
        return res.status(400).json({ error: 'Missing senderID or message' });
    }

    // ตรวจสอบว่าผู้ส่งมีสิทธิ์ส่งข้อความใน match นี้หรือไม่
    const checkUserInMatchQuery = `
        SELECT * FROM matches 
        WHERE matchID = ? AND (user1ID = ? OR user2ID = ?)
    `;

    pool.query(checkUserInMatchQuery, [matchID, senderID, senderID], (err, matchResults) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).json({ error: 'Database error' });
        }

        if (matchResults.length === 0) {
            return res.status(403).json({ error: 'User not authorized to send message in this chat' });
        }

        // ตรวจสอบสถานะการบล็อก
        const checkBlockQuery = `
            SELECT * FROM blocked_chats 
            WHERE matchID = ? AND isBlocked = 1
        `;

        pool.query(checkBlockQuery, [matchID], (err, blockResults) => {
            if (err) {
                console.error('Database error:', err);
                return res.status(500).json({ error: 'Database error' });
            }

            if (blockResults.length > 0) {
                return res.status(403).json({ error: 'Cannot send message. This chat has been blocked.' });
            }

            // บันทึกข้อความ
            const insertChatQuery = `
                INSERT INTO chats (matchID, senderID, message, timestamp)
                VALUES (?, ?, ?, NOW())
            `;

            pool.query(insertChatQuery, [matchID, senderID, message], (err, result) => {
                if (err) {
                    console.error('Database error:', err);
                    return res.status(500).json({ error: 'Database error' });
                }
                
                res.status(200).json({ 
                    success: 'Message sent successfully',
                    messageID: result.insertId
                });
            });
        });
    });
});


// API Delete Chat (ซ่อน chat ฝั่งเดียว)
app.post("/api/delete-chat", (req, res) => {
    const { userID, matchID } = req.body;

    if (!userID || !matchID) {
        return res.status(400).json({ error: 'Missing userID or matchID' });
    }

    const deleteQuery = `
        INSERT INTO deleted_chats (userID, matchID, deleted, deleteTimestamp)
        VALUES (?, ?, 1, NOW())
        ON DUPLICATE KEY UPDATE deleted = 1, deleteTimestamp = NOW();
    `;

    pool.query(deleteQuery, [userID, matchID], (err, result) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).json({ error: 'Database error' });
        }
        res.status(200).json({ success: 'Chat deleted successfully' });
    });
});


// API Restore All Chats
app.post("/api/restore-all-chats", (req, res) => {
    const { userID } = req.body;

    if (!userID) {
        return res.status(400).json({ error: 'Missing userID' });
    }

    const restoreAllQuery = `
        DELETE FROM deleted_chats
        WHERE userID = ?;
    `;

    pool.query(restoreAllQuery, [userID], (err, result) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).json({ error: 'Database error' });
        }
        res.status(200).json({ 
            success: 'All chats restored successfully',
            affectedChats: result.affectedRows
        });
    });
});


// API Block Chat
app.post("/api/block-chat", (req, res) => {
    const { userID, matchID, isBlocked } = req.body;

    if (!userID || !matchID || isBlocked === undefined) {
        return res.status(400).json({ error: 'Missing userID, matchID, or isBlocked' });
    }

    // ตรวจสอบว่า user มีสิทธิ์ block chat นี้หรือไม่
    const matchQuery = `SELECT user1ID, user2ID FROM matches WHERE matchID = ?`;
    
    pool.query(matchQuery, [matchID], (err, results) => {
        if (err || results.length === 0) {
            console.error('Database error or match not found');
            return res.status(500).json({ error: 'Match not found or database error' });
        }

        const { user1ID, user2ID } = results[0];
        
        // ตรวจสอบว่า userID เป็นหนึ่งในผู้ใช้ใน match นี้
        if (userID != user1ID && userID != user2ID) {
            return res.status(403).json({ error: 'User not authorized to block this chat' });
        }

        // กำหนดว่าใครเป็น blocker และใครถูก block
        const blockerID = userID;
        const blockedID = (userID == user1ID) ? user2ID : user1ID;

        // ตรวจสอบว่ามี block record อยู่แล้วหรือไม่
        const checkQuery = `SELECT blockID FROM blocked_chats WHERE matchID = ? AND user1ID = ?`;
        
        pool.query(checkQuery, [matchID, blockerID], (err, checkResult) => {
            if (err) {
                console.error('Database error:', err);
                return res.status(500).json({ error: 'Database error' });
            }

            if (checkResult.length > 0) {
                // อัปเดต block ที่มีอยู่
                const updateQuery = `
                    UPDATE blocked_chats 
                    SET isBlocked = ?, blockTimestamp = NOW() 
                    WHERE matchID = ? AND user1ID = ?`;
                    
                pool.query(updateQuery, [isBlocked ? 1 : 0, matchID, blockerID], (err, result) => {
                    if (err) {
                        console.error('Database error:', err);
                        return res.status(500).json({ error: 'Database error' });
                    }
                    res.status(200).json({ 
                        success: isBlocked ? 'Chat blocked successfully' : 'Chat unblocked successfully' 
                    });
                });
            } else {
                // สร้าง block record ใหม่
                const insertQuery = `
                    INSERT INTO blocked_chats (user1ID, user2ID, matchID, isBlocked, blockTimestamp)
                    VALUES (?, ?, ?, ?, NOW())`;
                    
                pool.query(insertQuery, [blockerID, blockedID, matchID, isBlocked ? 1 : 0], (err, result) => {
                    if (err) {
                        console.error('Database error:', err);
                        return res.status(500).json({ error: 'Database error' });
                    }
                    res.status(200).json({ success: 'Chat blocked successfully' });
                });
            }
        });
    });
});


// API Unblock Chat
app.post("/api/unblock-chat", (req, res) => {
    const { userID, matchID } = req.body;

    if (!userID || !matchID) {
        return res.status(400).json({ error: 'Missing userID or matchID' });
    }

    // ปลดบล็อกโดยตั้งค่า isBlocked = 0
    const unblockQuery = `
        UPDATE blocked_chats 
        SET isBlocked = 0, blockTimestamp = NOW()
        WHERE matchID = ? AND user1ID = ?;
    `;

    pool.query(unblockQuery, [matchID, userID], (err, result) => {
        if (err) {
            console.error("Database error:", err);
            return res.status(500).json({ error: 'Database error' });
        }

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: 'No block record found to unblock' });
        }

        res.status(200).json({ success: 'Chat unblocked successfully' });
    });
});


//check block status
app.post("/api/check-block-status", (req, res) => {
    const { matchID, userID } = req.body;
    
    const query = `
        SELECT 
            CASE WHEN EXISTS (
                SELECT 1 FROM blocked_chats 
                WHERE matchID = ? AND user1ID = ? AND isBlocked = 1
            ) THEN 1 ELSE 0 END as blockedByMe,
            CASE WHEN EXISTS (
                SELECT 1 FROM blocked_chats 
                WHERE matchID = ? AND user2ID = ? AND isBlocked = 1
            ) THEN 1 ELSE 0 END as blockedByOther
    `;
    
    pool.query(query, [matchID, userID, matchID, userID], (err, results) => {
        if (err) {
            return res.status(500).json({ error: 'Database error' });
        }
        
        res.json({
            blockedByMe: results[0].blockedByMe === 1,
            blockedByOther: results[0].blockedByOther === 1
        });
    });
});


//########################################################   Order API  ########################################################


// POST /api/orders
app.post("/api/orders", (req, res) => {
  console.log('[INFO] Received POST /api/orders request');
  const { user_id, package_id, title, content, link, image, prompay_number, ad_start_date } = req.body; // Add prompay_number
  if (!user_id || !package_id || !title || !content || !prompay_number) { // Make prompay_number mandatory
    console.warn('[WARN] Missing required fields for order creation.');
    return res.status(400).json({ error: 'Missing required fields (user_id, package_id, title, content, prompay_number)' });
  }
  // ตรวจสอบ ad_start_date
  if (!ad_start_date) {
    return res.status(400).json({ error: 'กรุณาเลือกวันที่ต้องการลงโฆษณา' });
  }
  const today = new Date();
  const minDate = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 2);
  const userDate = new Date(ad_start_date);

  if (userDate < minDate) {
    return res.status(400).json({ error: 'วันที่เริ่มโฆษณาต้องถัดจากวันนี้อย่างน้อย 2 วัน' });
  }
  // ดึงข้อมูล package
  pool.query('SELECT * FROM ad_packages WHERE package_id = ?', [package_id], (err, pkg) => {
    if (err) {
      console.error('[ERROR] Database error fetching package:', err);
      return res.status(500).json({ error: 'Database error' });
    }
    if (pkg.length === 0) {
      console.warn(`[WARN] Invalid package_id: ${package_id}`);
      return res.status(400).json({ error: 'Invalid package' });
    }
    const amount = pkg[0].price;
    const duration = pkg[0].duration_days;
    // สร้าง order
    // เพิ่ม prompay_number ใน SQL query และ VALUES
    const sql = `
          INSERT INTO orders (user_id, amount, status, created_at, updated_at, prompay_number)
          VALUES (?, ?, 'pending', NOW(), NOW(), ?)
      `;
    pool.query(sql, [user_id, amount, prompay_number], (err, result) => { // เพิ่ม prompay_number ที่นี่
      if (err) {
        console.error('[ERROR] Database error creating order:', err);
        return res.status(500).json({ error: 'Database error' });
      }
      const order_id = result.insertId;
      console.log(`[INFO] Order ID ${order_id} created with status 'pending'.`);
      // สร้างโฆษณาแบบ pending (รอจ่ายเงิน)
      const adSql = `
            INSERT INTO ads (user_id, order_id, title, content, link, image, status, show_at, created_at, expiration_date)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, NOW(), DATE_ADD(?, INTERVAL ? DAY))
        `;
      pool.query(adSql, [user_id, order_id, title, content, link || '', image || '', ad_start_date, ad_start_date, duration], (err2) => {
        if (err2) {
          console.error('[ERROR] Database error creating ad for order ID ' + order_id + ':', err2);
          return res.status(500).json({ error: 'Database error (ads)' });
        }
        console.log(`[INFO] Ad created for Order ID ${order_id} with status 'pending'.`);
        res.status(201).json({ order_id, amount, duration });
      });
    });
  });
});


// GET /api/orders/:orderId - สำหรับ user ดูข้อมูลออเดอร์
app.get("/api/orders/:orderId", (req, res) => {
  const { orderId } = req.params;
  console.log(`[INFO] Received GET /api/orders/${orderId} request`);
  const sql = `
      SELECT o.*, a.title, a.content, a.link, a.image, a.status AS ad_status
      FROM orders o
      LEFT JOIN ads a ON o.id = a.order_id
      WHERE o.id = ?
  `;
  pool.query(sql, [orderId], (err, results) => {
      if (err) {
          console.error(`[ERROR] Database error fetching order ${orderId}:`, err);
          return res.status(500).json({ error: 'Database error' });
      }
      if (results.length === 0) {
          console.warn(`[WARN] Order ID ${orderId} not found.`);
          return res.status(404).json({ error: 'Order not found' });
      }
      console.log(`[INFO] Order ID ${orderId} fetched successfully.`);
      res.json(results[0]);
  });
});


// POST /api/orders/:orderId/upload-slip
app.post("/api/orders/:orderId/upload-slip", upload.single('slip_image'), (req, res) => {
  const { orderId } = req.params;
  console.log(`[INFO] Received POST /api/orders/${orderId}/upload-slip request.`);

  if (!req.file) {
      console.warn(`[WARN] No slip_image file uploaded for order ID ${orderId}.`);
      return res.status(400).json({ error: 'กรุณาอัปโหลดสลิปการโอนเงิน' });
  }

  pool.getConnection((err, connection) => {
      if (err) {
          console.error('[ERROR] Database connection error:', err);
          return res.status(500).json({ error: 'Database connection error' });
      }

      connection.beginTransaction(transactionErr => {
          if (transactionErr) {
              connection.release();
              console.error('[ERROR] Transaction begin error:', transactionErr);
              return res.status(500).json({ error: 'Transaction error' });
          }

          // เพิ่ม show_at ในการ SELECT ด้วย เพราะมันถูกบันทึกไว้ใน orders ตอนสร้าง renew order แล้ว
          const getOrderSql = 'SELECT status, renew_ads_id, package_id, show_at FROM orders WHERE id = ?';
          connection.query(getOrderSql, [orderId], (orderErr, orderResults) => {
              if (orderErr) {
                  return connection.rollback(() => {
                      connection.release();
                      console.error(`[ERROR] Database error checking order status for order ${orderId}:`, orderErr);
                      res.status(500).json({ error: 'เกิดข้อผิดพลาดในการตรวจสอบสถานะออเดอร์' });
                  });
              }
              if (orderResults.length === 0) {
                  return connection.rollback(() => {
                      connection.release();
                      res.status(404).json({ error: 'ไม่พบออเดอร์นี้' });
                  });
              }

              const orderStatus = orderResults[0].status;
              const renewAdsId = orderResults[0].renew_ads_id;
              const packageId = orderResults[0].package_id;
              // const adShowAtFromOrder = orderResults[0].show_at; // ไม่ได้ใช้ในการอัปเดต ad โดยตรง แต่เก็บไว้ดูได้

              let canUpload = false;
              if (renewAdsId !== null) {
                  // สำหรับ Order ต่ออายุ ตรวจสอบสถานะ Ad เดิม และวันที่หมดอายุ
                  const getAdForRenewalSql = 'SELECT status, expiration_date, show_at FROM ads WHERE id = ?'; // ดึง show_at มาด้วย
                  connection.query(getAdForRenewalSql, [renewAdsId], (adErr, adResults) => {
                      if (adErr) {
                          return connection.rollback(() => {
                              connection.release();
                              console.error(`[ERROR] Database error checking ad status for renewal ad ${renewAdsId}:`, adErr);
                              res.status(500).json({ error: 'เกิดข้อผิดพลาดในการตรวจสอบสถานะโฆษณา' });
                          });
                      }
                      if (adResults.length === 0) {
                          return connection.rollback(() => {
                              connection.release();
                              res.status(404).json({ error: 'ไม่พบโฆษณาที่ต้องการต่ออายุ' });
                          });
                      }

                      const currentAdStatus = adResults[0].status;
                      const adExpirationDate = new Date(adResults[0].expiration_date);
                      const adOriginalShowAt = adResults[0].show_at; // show_at เดิมของ Ad

                      const today = new Date();
                      today.setHours(0, 0, 0, 0); // ตัดเวลาออก

                      // ตรวจสอบสถานะ Order ต้องเป็น 'pending' และ Ad เดิมต้องยังไม่หมดอายุ
                      if (orderStatus === 'pending' && adExpirationDate >= today) {
                          canUpload = true;
                      } else {
                          console.warn(`[WARN] Cannot upload slip for renewal order ${orderId}. Order status: ${orderStatus}, Ad expiration: ${adExpirationDate.toISOString().split('T')[0]}, Ad status: ${currentAdStatus}`);
                          let errorMessage = 'ไม่สามารถอัปโหลดสลิปได้ สถานะออเดอร์ไม่ถูกต้อง';
                          if (adExpirationDate < today) {
                              errorMessage = 'ไม่สามารถต่ออายุได้ เนื่องจากโฆษณาหมดอายุแล้ว';
                          }
                          return connection.rollback(() => {
                              connection.release();
                              res.status(400).json({ error: errorMessage });
                          });
                      }

                      // ส่ง originalExpirationDate และ originalShowAt เข้าไปในฟังก์ชัน helper
                      proceedUpdateOrderAndAd(connection, orderId, req.file.path, renewAdsId, packageId, adExpirationDate, adOriginalShowAt, canUpload, res);
                  });
              } else {
                  // Logic เดิมสำหรับโฆษณาใหม่ (ไม่ต้องมี originalExpirationDate และ originalShowAt)
                  if (orderStatus === 'approved') {
                      canUpload = true;
                  } else {
                      console.warn(`[WARN] Cannot upload slip for new ad order ${orderId}. Current status: ${orderStatus}`);
                      return connection.rollback(() => {
                          connection.release();
                          res.status(400).json({ error: 'ไม่สามารถอัปโหลดสลิปได้ ต้องรอให้แอดมินอนุมัติเนื้อหาก่อน' });
                      });
                  }
                  proceedUpdateOrderAndAd(connection, orderId, req.file.path, renewAdsId, packageId, null, null, canUpload, res); // ส่ง null สำหรับค่าที่ไม่เกี่ยวข้อง
              }
          });
      });
  });
});


//########################################################   Ads API  ########################################################


// GET /api/ad-packages
app.get("/api/ad-packages", (req, res) => {
  console.log('[INFO] Received GET /api/ad-packages request');
  const sql = 'SELECT * FROM ad_packages ORDER BY duration_days ASC';
  pool.query(sql, (err, results) => {
      if (err) {
          console.error('[ERROR] Database error fetching ad packages:', err);
          return res.status(500).json({ error: 'Database error' });
      }
      console.log(`[INFO] Fetched ${results.length} ad packages.`);
      res.json(results);
  });
});


// GET /api/user/ads - สำหรับ user ดูโฆษณาของตัวเอง
app.get("/api/user/ads", authenticateToken, (req, res) => {
  const userId = req.user.id; // สมมติว่า authenticateToken ใส่ user id ใน req.user
  const sql = `
    SELECT title, content, link, image, status, show_at, expiration_date, admin_notes, admin_slip
    FROM ads
    WHERE user_id = ?
    ORDER BY created_at DESC
  `;
  pool.query(sql, [userId], (err, results) => {
    if (err) {
      console.error('[ERROR] Database error fetching user ads:', err);
      return res.status(500).json({ error: 'Database error' });
    }
    res.json(results);
  });
});


// POST /api/ads/:adId/renew - สำหรับต่ออายุโฆษณา
app.post("/api/ads/:adId/renew", authenticateToken, (req, res) => {
  const { adId } = req.params;
  const { package_id, prompay_number } = req.body;
  const user_id = req.user.id;

  console.log(`[INFO] Received POST /api/ads/${adId}/renew request from user ${user_id}`);

  if (!package_id || !prompay_number) {
      console.warn('[WARN] Missing required fields for ad renewal.');
      return res.status(400).json({ error: 'กรุณากรอกข้อมูลให้ครบถ้วน (package_id, prompay_number)' });
  }

  pool.getConnection((err, connection) => {
      if (err) {
          console.error('[ERROR] Database connection error:', err);
          return res.status(500).json({ error: 'Database connection error' });
      }

      connection.beginTransaction(transactionErr => {
          if (transactionErr) {
              connection.release();
              console.error('[ERROR] Transaction begin error:', transactionErr);
              return res.status(500).json({ error: 'Transaction error' });
          }

          // --- เพิ่ม Logic ตรวจสอบและลบ Order เก่าที่ค้างชำระเกิน 2 ชั่วโมง (สำหรับ renew_ads_id เดียวกัน) ---
          // Current Time (Bangkok)
          const currentTime = new Date(); // เวลาปัจจุบัน
          // Calculate 2 hours ago from current time
          const twoHoursAgo = new Date(currentTime.getTime() - 2 * 60 * 60 * 1000); 
          // Format for MySQL DATETIME (e.g., '2025-07-23 21:09:23')
          const twoHoursAgoISO = twoHoursAgo.toISOString().slice(0, 19).replace('T', ' '); 

          const deleteExpiredPendingOrdersSql = `
              DELETE FROM orders
              WHERE user_id = ?
              AND renew_ads_id = ?
              AND status = 'pending'
              AND created_at < ?;
          `;

          connection.query(deleteExpiredPendingOrdersSql, [user_id, adId, twoHoursAgoISO], (deleteErr, deleteResult) => {
              if (deleteErr) {
                  return connection.rollback(() => {
                      connection.release();
                      console.error(`[ERROR] Database error deleting expired pending orders for ad ${adId}:`, deleteErr);
                      res.status(500).json({ error: 'เกิดข้อผิดพลาดในการลบคำสั่งซื้อเก่า' });
                  });
              }
              if (deleteResult.affectedRows > 0) {
                  console.log(`[INFO] Deleted ${deleteResult.affectedRows} expired pending renewal orders for ad ${adId} by user ${user_id}.`);
              }

              // ดึง expiration_date และ show_at มาด้วย เพื่อตรวจสอบและใช้เป็น show_at ของ Order ใหม่
              const getAdSql = 'SELECT user_id, title, content, link, image, expiration_date, show_at FROM ads WHERE id = ? AND user_id = ?';
              connection.query(getAdSql, [adId, user_id], (adErr, adResults) => {
                  if (adErr) {
                      return connection.rollback(() => {
                          connection.release();
                          console.error(`[ERROR] Database error fetching ad ${adId}:`, adErr);
                          res.status(500).json({ error: 'Database error fetching ad' });
                      });
                  }
                  if (adResults.length === 0) {
                      return connection.rollback(() => {
                          connection.release();
                          console.warn(`[WARN] Ad ID ${adId} not found or not owned by user ${user_id}.`);
                          res.status(404).json({ error: 'ไม่พบโฆษณาหรือคุณไม่มีสิทธิ์ต่ออายุโฆษณานี้' });
                      });
                  }

                  const adExpirationDate = new Date(adResults[0].expiration_date);
                  // ตั้งค่าเวลาให้เป็น 00:00:00 ของ Local Timezone เพื่อให้การเปรียบเทียบวันถูกต้อง
                  // และเพื่อเตรียมพร้อมสำหรับการบวกวันโดยไม่ให้ Timezone มีผลกระทบกับวัน
                  adExpirationDate.setHours(0, 0, 0, 0);

                  const today = new Date();
                  today.setHours(0, 0, 0, 0); // ตัดเวลาออก เพื่อเปรียบเทียบแค่ในระดับวัน

                  // เพิ่มการตรวจสอบ: โฆษณาต้องยังไม่หมดอายุจึงจะต่อได้
                  if (adExpirationDate < today) {
                      console.warn(`[WARN] Ad ID ${adId} has expired. Cannot renew.`);
                      return connection.rollback(() => {
                          connection.release();
                          res.status(400).json({ error: 'ไม่สามารถต่ออายุได้ เนื่องจากโฆษณาหมดอายุแล้ว' });
                      });
                  }

                  // กำหนด show_at ของ Order ใหม่ให้เป็น วันที่หมดอายุของ Ad เดิม + 1 วัน
                  // เพื่อให้การต่ออายุเริ่มนับจากวันถัดไปหลังจากโฆษณาเดิมหมดอายุ
                  const nextDayAfterExpiration = new Date(adExpirationDate);
                  nextDayAfterExpiration.setDate(adExpirationDate.getDate() + 1); // บวกไป 1 วัน

                  // Format เป็น YYYY-MM-DD โดยใช้ ISO string แล้วตัดส่วนเวลาออก (จะใช้ UTC แต่ไม่เป็นปัญหาเมื่อใช้แค่ส่วนวันที่)
                  const show_at_for_order = nextDayAfterExpiration.toISOString().split('T')[0];

                  console.log(`[INFO] Ad Expiration Date: ${adExpirationDate.toISOString().split('T')[0]}, Next day for Order's show_at: ${show_at_for_order}`);

                  const getPkgSql = 'SELECT price, duration_days FROM ad_packages WHERE package_id = ?';
                  connection.query(getPkgSql, [package_id], (pkgErr, pkgResults) => {
                      if (pkgErr) {
                          return connection.rollback(() => {
                              connection.release();
                              console.error('[ERROR] Database error fetching ad package:', pkgErr);
                              res.status(500).json({ error: 'Database error fetching package' });
                          });
                      }
                      if (pkgResults.length === 0) {
                          return connection.rollback(() => {
                              connection.release();
                              console.warn(`[WARN] Invalid package_id: ${package_id}`);
                              res.status(400).json({ error: 'ไม่พบแพ็กเกจที่เลือก' });
                          });
                      }

                      const { price, duration_days } = pkgResults[0];

                      const insertOrderSql = `
                          INSERT INTO orders (user_id, amount, status, prompay_number, renew_ads_id, package_id, show_at, created_at, updated_at)
                          VALUES (?, ?, 'pending', ?, ?, ?, ?, NOW(), NOW())
                      `;
                      // ใช้ show_at_for_order ที่คำนวณจาก expiration_date ของ Ad เดิม + 1 วัน
                      connection.query(insertOrderSql, [user_id, price, prompay_number, adId, package_id, show_at_for_order], (orderErr, orderResult) => {
                          if (orderErr) {
                              return connection.rollback(() => {
                                  connection.release();
                                  console.error('[ERROR] Database error creating renewal order:', orderErr);
                                  res.status(500).json({ error: 'เกิดข้อผิดพลาดในการสร้างคำสั่งซื้อต่ออายุ' });
                              });
                          }

                          const order_id = orderResult.insertId;
                          console.log(`[INFO] Renewal Order ID ${order_id} created for ad ${adId} with status 'pending'.`);

                          connection.commit(commitErr => {
                              if (commitErr) {
                                  return connection.rollback(() => {
                                      connection.release();
                                      console.error('[ERROR] Transaction commit error:', commitErr);
                                      res.status(500).json({ error: 'Transaction commit error' });
                                  });
                              }
                              connection.release();
                              res.status(201).json({
                                  message: 'สร้างคำสั่งซื้อต่ออายุสำเร็จ! กรุณาชำระเงินและอัปโหลดสลิป',
                                  order_id,
                                  amount: price,
                                  duration: duration_days,
                                  show_at: show_at_for_order, // ส่ง show_at ที่ใช้จริงกลับไป
                              });
                          });
                      });
                  });
              });
          }); // ปิด deleteErr callback
      });
  });
});


//########################################################   End API  ########################################################


// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});