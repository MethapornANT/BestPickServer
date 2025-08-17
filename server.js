// ======================================================
// Core setup: env, libs, app
// ======================================================
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

// ======================================================
// Middleware: request parsing & CORS
// ======================================================
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());

// ======================================================
// Firebase Admin: initialize credentials
// ======================================================
const serviceAccount = require("./config/apilogin-6efd6-firebase-adminsdk-b3l6z-c2e5fe541a.json");
const { title } = require("process");
const { error } = require("console");
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

// ======================================================
// MySQL: connection pool with SSL
// ======================================================
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
    rejectUnauthorized: false,
    ca: fs.readFileSync("./certs/isrgrootx1.pem"),
  },
});

// ======================================================
// DB: lightweight auto-reconnect (non-invasive)
// ======================================================
function reconnect() {
  pool.getConnection((err) => {
    if (err) {
      console.error("Error re-establishing database connection:", err);
      setTimeout(reconnect, 2000);
    } else {
      console.log("Database reconnected successfully.");
    }
  });
}

// ======================================================
// DB: connection error handling
// ======================================================
pool.on('error', (err) => {
  if (err.code === 'PROTOCOL_CONNECTION_LOST' || err.code === 'ECONNRESET') {
    console.error("Database connection lost. Reconnecting...");
    reconnect();
  } else {
    console.error("Database error:", err);
    throw err;
  }
});

// ======================================================
// DB: initial connectivity check
// ======================================================
pool.getConnection((err, connection) => {
  if (err) {
    console.error("Error connecting to the database:", err);
    return;
  }
  console.log("Connected to the database successfully!");
  connection.release();
});

// ======================================================
// Export: pool for use in other modules
// ======================================================
module.exports = pool;

// ======================================================
// Auth: Verify token (attach id, role) – used where needed
// ======================================================
const verifyToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(403).json({ error: "No token provided or incorrect format" });
  }

  const token = authHeader.split(" ")[1];
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.userId = decoded.id;
    req.role = decoded.role;
    next();
  } catch (err) {
    return res.status(401).json({ error: "Unauthorized: Invalid token" });
  }
};

// ======================================================
// Uploads: disk storage with unique filename
// ======================================================
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
    const fileExtension = path.extname(file.originalname);
    const originalName = path.basename(file.originalname, fileExtension);
    const timestamp = Date.now();
    const newFileName = `${timestamp}_${originalName}_${uniqueName}${fileExtension}`;
    console.log(`File saved as: ${newFileName}`);
    cb(null, newFileName);
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 2147483648 },
});

// ======================================================
// OTP: generate 4-digit code (hex-based, trimmed)
// ======================================================
function generateOtp() {
  const otp = crypto.randomBytes(3).toString("hex");
  return parseInt(otp, 16).toString().slice(0, 4);
}

// ======================================================
// Email: send OTP via Gmail SMTP (callback-style)
// ======================================================
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
        <p style="margin-top: 20px;">Thanks, <br> BestPick Team</p>
        <hr>
        <p style="font-size: 12px; color: #999;">This is an automated email, please do not reply.</p>
      </div>
    `,
  };

  transporter.sendMail(mailOptions, (error, info) => {
    if (error) {
      console.error("Error sending OTP email:", error);
      return callback({ error: "Failed to send OTP email. Please try again later." });
    }
    callback(null, info);
  });
}

// ======================================================
// Auth: JWT check (attach payload to req.user)
// ======================================================
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (token == null) {
    console.log('Access Denied: No token provided.');
    return res.sendStatus(401);
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      console.log('Forbidden: Invalid token.', err.message);
      return res.sendStatus(403);
    }
    req.user = user;
    next();
  });
};

// ======================================================
// AuthZ: admin-only guard (expects req.user.role)
// ======================================================
const authorizeAdmin = (req, res, next) => {
  if (req.user && req.user.role === 'admin') {
    next();
  } else {
    console.log('Forbidden: Admin access required. User role:', req.user ? req.user.role : 'N/A');
    res.status(403).json({ message: "Forbidden: Admin access required" });
  }
};



//########################################################   Register API  #######################################################


// ======================================================
// Auth: Registration via Email (OTP flow)
// ======================================================
app.post("/api/register/email", async (req, res) => {
  try {
    const { email } = req.body;

    // CHECK: already registered & active (email+password exists)
    const checkRegisteredSql =
      "SELECT * FROM users WHERE email = ? AND status = 'active' AND password IS NOT NULL";

    pool.query(checkRegisteredSql, [email], (err, results) => {
      if (err) throw new Error("Database error during email registration check");

      if (results.length > 0)
        return res.status(400).json({ error: "Email already registered" });

      // CHECK: exists but deactivated -> reactivate
      const checkDeactivatedSql =
        "SELECT * FROM users WHERE email = ? AND status = 'deactivated'";

      pool.query(checkDeactivatedSql, [email], (err, deactivatedResults) => {
        if (err) throw new Error("Database error during email check");

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
          // CHECK: email in use but registration incomplete (no password yet)
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

            // OTP: issue new or update existing
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
                  // SECURITY: OTP goes to email only; do not log OTP
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
                  // SECURITY: OTP goes to email only; do not log OTP
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
    // ERROR: fallback (note throws in callbacks won't be caught here)
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Auth: Verify registration OTP
// ======================================================
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
    // ERROR: fallback
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Auth: Set password (complete registration)
// ======================================================
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
    // ERROR: fallback
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Auth: Resend OTP (registration)
// ======================================================
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
        // ISSUE: expired -> generate new OTP
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
        // REUSE: still valid -> resend same OTP
        sendOtpEmail(email, otp, (error) => {
          if (error) throw new Error("Error resending OTP email");
          res.status(200).json({ message: "OTP resent to email" });
        });
      }
    });
  } catch (error) {
    // ERROR: fallback
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Auth: Forgot password (issue reset OTP)
// ======================================================
app.post("/api/forgot-password", async (req, res) => {
  try {
    const { email } = req.body;
    const userCheckSql =
      "SELECT * FROM users WHERE email = ? AND password IS NOT NULL AND status = 'active'";

    pool.query(userCheckSql, [email], (err, userResults) => {
      if (err) throw new Error("Database error during email check");
      if (userResults.length === 0)
        return res.status(400).json({ error: "Email not found" });

      // OTP: create or update for password reset
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
    // ERROR: fallback
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Auth: Verify reset OTP
// ======================================================
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
    // ERROR: fallback
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Auth: Reset password (apply new password)
// ======================================================
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
    // ERROR: fallback
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Auth: Resend OTP for reset password
// ======================================================
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
    // ERROR: fallback
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});



//########################################################   Login API  #######################################################


// ======================================================
// Auth: Login (email/password + Google) with basic lockout
// ======================================================
app.post("/api/login", async (req, res) => {
  try {
      const { email, password, google_id } = req.body; // รับ google_id เพิ่มเข้ามาด้วย

      // INFO: Client IP (เพื่อบันทึก last_login_ip)
      const ipAddress = req.headers["x-forwarded-for"] || req.connection.remoteAddress;

      // VALIDATION: ต้องมี email เสมอ
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

          // --------------------------------------------------
          // Google login branch
          // --------------------------------------------------
          if (google_id) {
              if (!user) {
                  // CREATE: ผู้ใช้ใหม่จาก Google (ไม่มี record เดิม)
                  const insertSql = "INSERT INTO users (email, google_id, created_at, last_login, last_login_ip, status, role, failed_attempts) VALUES (?, ?, NOW(), NOW(), ?, ?, ?, ?)";
                  pool.query(insertSql, [email, google_id, ipAddress, 'active', 'user', 0], (insertErr, insertResult) => {
                      if (insertErr) {
                          console.error("Error creating new user with Google ID:", insertErr);
                          return res.status(500).json({ message: "Failed to create user with Google ID." });
                      }
                      user = {
                          id: insertResult.insertId,
                          email: email,
                          username: null,
                          picture: null,
                          google_id: google_id,
                          password: null,
                          status: 'active',
                          role: 'user',
                          failed_attempts: 0,
                          last_login: new Date(),
                          last_login_ip: ipAddress,
                      };
                      // TOKEN: ออก JWT ให้ทันที
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
                  return; // จบ branch นี้
              }

              // LINK/LOGIN: พบ user จาก email
              if (user.google_id === null) {
                  // LINK: ผูก google_id ให้บัญชีที่เคยสมัครแบบรหัสผ่าน
                  const updateGoogleIdSql = "UPDATE users SET google_id = ? WHERE id = ?";
                  pool.query(updateGoogleIdSql, [google_id, user.id], (updateErr) => {
                      if (updateErr) {
                          console.error("Error updating Google ID:", updateErr);
                          return res.status(500).json({ message: "Failed to link Google account." });
                      }
                      user.google_id = google_id;
                      // SUCCESS: เข้าสู่ระบบเหมือนเดิม
                      handleSuccessfulLogin(res, user, ipAddress);
                  });
              } else if (user.google_id === google_id) {
                  // LOGIN: google_id ตรงกัน
                  handleSuccessfulLogin(res, user, ipAddress);
              } else {
                  // CONFLICT: email นี้เคยผูกกูเกิลอีกบัญชี
                  return res.status(400).json({ message: "This email is already associated with a different Google account." });
              }
          }

          // --------------------------------------------------
          // Email/Password login branch
          // --------------------------------------------------
          else if (password) {
              if (!user) {
                  return res.status(404).json({ message: "No user found with this email." });
              }

              // CHECK: บัญชีที่เคยสมัครด้วย Google-only (ไม่มีรหัสผ่าน)
              if (user.password === null) {
                  return res.status(400).json({ message: "Please sign in using Google or set a password for this account first." });
              }

              // CHECK: สถานะต้อง active
              if (user.status !== 'active') {
                  return res.status(403).json({ message: "User is Suspended" });
              }

              // RATE-LIMIT (เบื้องต้น): บล็อกชั่วคราวเมื่อพยายามผิดซ้ำ
              if (user.failed_attempts >= 5 && user.last_failed_attempt) {
                  const now = Date.now();
                  const timeSinceLastAttempt = now - new Date(user.last_failed_attempt).getTime();
                  if (timeSinceLastAttempt < 300000) { // 5 นาที
                      return res.status(429).json({
                          message: "Too many failed login attempts. Try again in 5 minutes.",
                      });
                  }
              }

              // VERIFY: ตรวจรหัสผ่าน
              bcrypt.compare(password, user.password, (err, isMatch) => {
                  if (err) {
                      console.error("Password comparison error:", err);
                      return res.status(500).json({ error: "Password comparison error" });
                  }
                  if (!isMatch) {
                      // LOG: บันทึกความพยายามล้มเหลว
                      const updateFailSql = "UPDATE users SET failed_attempts = failed_attempts + 1, last_failed_attempt = NOW() WHERE id = ?";
                      pool.query(updateFailSql, [user.id], (err) => {
                          if (err) console.error("Error logging failed login attempt:", err);
                      });
                      return res.status(401).json({ message: "Email or Password is incorrect." });
                  }

                  // SUCCESS
                  handleSuccessfulLogin(res, user, ipAddress);
              });
          }

          // --------------------------------------------------
          // Missing credentials
          // --------------------------------------------------
          else {
              return res.status(400).json({ message: "Missing login credentials (password or google_id)." });
          }
      });
  } catch (error) {
      // NOTE: error ใน callback จะไม่โดน try/catch ด้านนอก (พฤติกรรมเดิม)
      console.error("Internal error:", error.message);
      res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Helper: Apply login success effects + issue JWT
// ======================================================
function handleSuccessfulLogin(res, user, ipAddress) {
  const resetFailSql = "UPDATE users SET failed_attempts = 0, last_login = NOW(), last_login_ip = ? WHERE id = ?";
  pool.query(resetFailSql, [ipAddress, user.id], (err) => {
      if (err) {
          console.error("Error resetting failed attempts or updating login time:", err);
          return res.status(500).json({ error: "Error updating login status." });
      }

      // TOKEN: ออก JWT หลังอัปเดตสถานะ login
      const token = jwt.sign({ id: user.id, role: user.role }, JWT_SECRET);

      // RESPONSE: ส่งข้อมูลผู้ใช้แบบย่อ
      res.status(200).json({
          message: "Authentication successful",
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
}


// ======================================================
// Profile: set/update with basic validation
// ======================================================
app.post("/api/set-profile", verifyToken, upload.single('picture'), (req, res) => {
  const { newUsername, birthday, gender } = req.body; // <<-- เพิ่ม gender เข้ามา
  const userId = req.userId;
  const picture = req.file ? `/uploads/${req.file.filename}` : null; 

  // REQUIRED: ทุกฟิลด์ต้องมีเพื่อให้ตั้งโปรไฟล์สมบูรณ์
  if (!newUsername || !picture || !birthday || !gender) {
    return res.status(400).json({ message: "New username, picture, birthday, and gender are required" });
  }

  // PARSE: แปลงวันเกิด DD/MM/YYYY -> YYYY-MM-DD
  const birthdayParts = birthday.split('/');
  if (birthdayParts.length !== 3 || isNaN(parseInt(birthdayParts[0])) || isNaN(parseInt(birthdayParts[1])) || isNaN(parseInt(birthdayParts[2]))) {
    return res.status(400).json({ message: "Invalid birthday format. Please use DD/MM/YYYY" });
  }
  const formattedBirthday = `${birthdayParts[2]}-${birthdayParts[1]}-${birthdayParts[0]}`;

  // AGE: คำนวณอายุแบบพื้นฐาน (ผิดพลาดให้เป็น 0)
  let age = null;
  try {
    const birthDate = new Date(formattedBirthday);
    const today = new Date();
    age = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    if (age < 0) {
        age = 0;
    }
  } catch (e) {
    console.error("Error calculating age:", e);
    // ไม่บังคับ return error เพื่อคงพฤติกรรมเดิม
  }

  // ENUM: ตรวจค่าที่อนุญาตของ gender
  const allowedGenders = ['Male', 'Female', 'Other'];
  if (!allowedGenders.includes(gender)) {
    return res.status(400).json({ message: "Invalid gender value. Must be Male, Female, or Other." });
  }

  // UNIQUE: ตรวจซ้ำชื่อผู้ใช้
  const checkUsernameQuery = "SELECT * FROM users WHERE username = ?";
  pool.query(checkUsernameQuery, [newUsername], (err, results) => {
    if (err) {
      console.error("Error checking username: ", err);
      return res.status(500).json({ message: "Database error checking username" });
    }

    if (results.length > 0) {
      return res.status(400).json({ message: "Username already taken" });
    }

    // UPDATE: บันทึกข้อมูลโปรไฟล์ใหม่
    const updateProfileQuery = "UPDATE users SET username = ?, picture = ?, birthday = ?, gender = ?, age = ? WHERE id = ?";
    pool.query(updateProfileQuery, [newUsername, picture, formattedBirthday, gender, age, userId], (err) => {
      if (err) {
        console.error("Error updating profile: ", err);
        return res.status(500).json({ message: "Error updating profile" });
      }

      return res.status(200).json({ message: "Profile set/updated successfully" });
    });
  });
});


// ======================================================
// Google Sign-In: create/link/reactivate user
// ======================================================
app.post("/api/google-signin", async (req, res) => {
  try {
    const { googleId, email } = req.body;

    // REQUIRED: ต้องมีทั้งสองฟิลด์
    if (!googleId || !email) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    // FIND: ผู้ใช้จาก google_id ที่ active/deactivated
    const checkGoogleIdSql =
      "SELECT * FROM users WHERE google_id = ? AND (status = 'active' OR status = 'deactivated')";
    pool.query(checkGoogleIdSql, [googleId], (err, googleIdResults) => {
      if (err) {
        console.error("Original database error during Google ID check:", err);
        throw new Error("Database error during Google ID check");
      }

      if (googleIdResults.length > 0) {
        const user = googleIdResults[0];

        // REACTIVATE: หาก deactivated ให้กลับเป็น active และอัปเดต email
        if (user.status === "deactivated") {
          const reactivateSql = "UPDATE users SET status = 'active', email = ? WHERE google_id = ?";
          pool.query(reactivateSql, [email, googleId], (err) => {
            if (err) {
              console.error("Original database error during user reactivation:", err);
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
          // UPDATE: กรณี active อัปเดต email ให้ล่าสุด
          const updateSql = "UPDATE users SET email = ? WHERE google_id = ?";
          pool.query(updateSql, [email, googleId], (err) => {
            if (err) {
              console.error("Original database error during user update:", err);
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
        // CHECK: email นี้มี user active อยู่แล้วหรือไม่
        const checkEmailSql = "SELECT * FROM users WHERE email = ? AND status = 'active'";
        pool.query(checkEmailSql, [email], (err, emailResults) => {
          if (err) {
            console.error("Original database error during email check:", err);
            throw new Error("Database error during email check");
          }
          if (emailResults.length > 0) {
            // CONFLICT: อีเมลนี้ใช้กับบัญชีอื่นแล้ว
            return res.status(409).json({
              error: "Email already registered with another account",
            });
          }

          // CREATE: ผู้ใช้ใหม่จาก Google
          const insertSql =
            "INSERT INTO users (google_id, email, username, status, role) VALUES (?, ?, '', 'active', 'user')";
          pool.query(insertSql, [googleId, email], (err, result) => {
            // หมายเหตุ: บรรทัด 756 เดิมที่เจอ error ยังคงลอกรหัสไว้เหมือนเดิม
            if (err) {
              console.error("Original database error during user insertion:", err);
              throw new Error("Database error during user insertion");
            }

            const newUserId = result.insertId;
            const newUserSql = "SELECT * FROM users WHERE id = ?";
            pool.query(newUserSql, [newUserId], (err, newUserResults) => {
              if (err) {
                console.error("Original database error during new user fetch:", err);
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
    // NOTE: โยน error ใน callback จะไม่ถูกจับที่นี่ (ยังคงพฤติกรรมเดิม)
    console.error("Caught error in Google Sign-In API:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});



//########################################################   Interactions API  #######################################################


// ======================================================
// Interactions: Create new interaction (user action/comment)
// SECURITY: requires verifyToken
// ======================================================
app.post("/api/interactions", verifyToken, async (req, res) => {
  const { post_id, action_type, content } = req.body;
  const user_id = req.userId; // from token

  // VALIDATION: required fields
  const postIdValue = post_id ? post_id : null;
  if (!user_id || !action_type) {
    return res
      .status(400)
      .json({ error: "Missing required fields: user_id or action_type" });
  }

  // DB: insert interaction
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

    // --- เพิ่ม Notification เฉพาะกรณี bookmark/unbookmark (fire-and-forget) ---
    if (postIdValue) {
      if (action_type === "bookmark") {
        const checkSql = `
          SELECT id FROM notifications
          WHERE user_id = ? AND post_id = ? AND action_type = 'bookmark'
          LIMIT 1
        `;
        pool.query(checkSql, [user_id, postIdValue], (cErr, cRows) => {
          if (cErr) {
            console.error("Check bookmark noti error:", cErr);
          } else if (cRows.length === 0) {
            const notiInsert = `
              INSERT INTO notifications (user_id, post_id, action_type, content)
              VALUES (?, ?, 'bookmark', NULL)
            `;
            pool.query(notiInsert, [user_id, postIdValue], (nErr) => {
              if (nErr) console.error("Insert bookmark noti error:", nErr);
            });
          }
        });
      } else if (action_type === "unbookmark") {
        const notiDelete = `
          DELETE FROM notifications
          WHERE user_id = ? AND post_id = ? AND action_type = 'bookmark'
        `;
        pool.query(notiDelete, [user_id, postIdValue], (dErr) => {
          if (dErr) console.error("Delete bookmark noti error:", dErr);
        });
      }
    }

    res.status(201).json({
      message: "Interaction saved successfully",
      interaction_id: results.insertId,
    });
  });
});



// ======================================================
// Interactions: Fetch all interactions (joined with users, posts)
// SECURITY: requires verifyToken
// NOTE: returns all users' interactions (admin-like view)
// ======================================================
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


// ======================================================
// Interactions: Fetch interactions by userId (self-only)
// SECURITY: path :userId must match token userId
// ======================================================
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


// ======================================================
// Interactions: Delete by id (self-only)
// SECURITY: only owner (token user) can delete
// ======================================================
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


// ======================================================
// Interactions: Update by id (self-only)
// SECURITY: only owner (token user) can update
// ======================================================
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


// ======================================================
// Util: simple JSON validator
// ======================================================
function isValidJson(str) {
  try {
    JSON.parse(str);
    return true;
  } catch (e) {
    return false;
  }
}


// ======================================================
// Likes: check like status for a post by user
// SECURITY: userId in path must match token userId
// ======================================================
app.get("/api/checkLikeStatus/:postId/:userId", verifyToken, (req, res) => {
  const { postId, userId } = req.params;
  const user_id = req.userId;

  // AUTHZ: enforce self-only access
  if (user_id != userId) {
    return res
      .status(403)
      .json({ error: "Unauthorized access: User ID does not match" });
  }

  // VALIDATION: required params
  if (!postId || !userId) {
    return res
      .status(400)
      .json({ error: "Missing required parameters: postId or userId" });
  }

  // DB: check like existence
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

    const isLiked = results[0].isLiked > 0;
    res.json({ isLiked });
  });
});



//########################################################   Post API  #######################################################


// ======================================================
// Posts: list all active posts for authenticated user
// SECURITY: requires verifyToken; joins author info; flags is_liked per user
// ======================================================
app.get("/api/posts", verifyToken, (req, res) => {
  try {
    const userId = req.userId; // from token

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

      // NOTE: keep mapping shape as-is; do not coerce arrays/booleans differently
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
    // NOTE: errors thrown inside pool.query callback won't be caught here
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Posts: view single post with like/comment counts and comments
// SECURITY: requires verifyToken; returns is_liked per requesting user
// ======================================================
app.get("/api/posts/:id", verifyToken, (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.userId; // from token

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
      console.log("Post data fetched:", post); // diagnostic log as per original

      // NOTE: preserve original JSON parsing behavior and boolean coercion
      post.photo_url = isValidJson(post.photo_url)
        ? JSON.parse(post.photo_url)
        : [post.photo_url];
      post.video_url = isValidJson(post.video_url)
        ? JSON.parse(post.video_url)
        : [post.video_url];
      post.is_liked = post.is_liked > 0;

      pool.query(queryComments, [id], (err, commentResults) => {
        if (err) {
          console.error("Database error during comments retrieval:", err);
          return res
            .status(500)
            .json({ error: "Internal server error during comments retrieval" });
        }

        console.log("Comment data fetched:", commentResults); // diagnostic

        res.json({
          ...post,
          like_count: post.like_count,
          productName: post.ProductName,
          comment_count: post.comment_count,
          update: post.updated_at,
          is_liked: post.is_liked,
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


// ======================================================
// Posts: delete by id (owner only)
// SECURITY: user must own the post; cascades related notifications
// ======================================================
app.delete("/api/posts/:id", verifyToken, (req, res) => {
  const { id } = req.params;
  const user_id = req.userId; // from token

  const postCheckSql = "SELECT * FROM posts WHERE id = ? AND user_id = ?";
  pool.query(postCheckSql, [id, user_id], (postError, postResults) => {
      if (postError) {
          console.error("Database error during post check:", postError);
          return res.status(500).json({ error: "Database error during post check" });
      }
      if (postResults.length === 0) {
          return res.status(404).json({ error: "Post not found or you are not the owner" });
      }

      const deleteNotificationsSql = "DELETE FROM notifications WHERE post_id = ?";
      pool.query(deleteNotificationsSql, [id], (deleteNotificationError) => {
          if (deleteNotificationError) {
              console.error("Database error during notification deletion:", deleteNotificationError);
              return res.status(500).json({ error: "Database error during notification deletion" });
          }

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


// ======================================================
// Categories: list (simple fetch using pooled connection)
// ======================================================
app.get("/api/type", verifyToken, (req, res) => {
  const sqlQuery = "SELECT * FROM category";

  // NOTE: keep explicit getConnection/release pattern as original
  pool.getConnection((err, connection) => {
    if (err) {
      return res.status(500).json({ error: "Error connecting to the database" });
    }

    connection.query(sqlQuery, (err, result) => {
      connection.release();

      if (err) {
        return res.status(500).json({ error: "Database query failed" });
      }

      res.json(result);
    });
  });
});


// ======================================================
// Likes: toggle like/unlike for a post (self-guarded)
// SECURITY: token user must match body.user_id
// ======================================================
app.post("/api/posts/like/:id", verifyToken, (req, res) => {
  const { id } = req.params;        // post id
  const { user_id } = req.body;     // user id from body

  try {
    // AUTHZ: enforce self-only action
    if (parseInt(req.userId) !== parseInt(user_id)) {
      return res
        .status(403)
        .json({ error: "You are not authorized to like this post" });
    }

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
          // UNLIKE: remove like then return updated count
          const unlikeSql =
            "DELETE FROM likes WHERE post_id = ? AND user_id = ?";
          pool.query(unlikeSql, [id, user_id], (err) => {
            if (err) {
              console.error("Database error during unlike:", err);
              return res
                .status(500)
                .json({ error: "Database error during unlike" });
            }

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
          // LIKE: insert like then return updated count
          const likeSql = "INSERT INTO likes (post_id, user_id) VALUES (?, ?)";
          pool.query(likeSql, [id, user_id], (err) => {
            if (err) {
              console.error("Database error during like:", err);
              return res
                .status(500)
                .json({ error: "Database error during like" });
            }

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
    // NOTE: try/catch here only captures sync errors
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// ======================================================
// Static: serve user-uploaded files (images/videos)
// ======================================================
app.use("/uploads", express.static(path.join(__dirname, "uploads")));


// ======================================================
// Search: grouped by username; basic text match on users/posts
// ======================================================
app.get("/api/search", (req, res) => {
  const { query } = req.query;

  // VALIDATION: require query
  if (!query) {
    return res.status(400).json({ error: "Search query is required" });
  }

  // NORMALIZE: trim + lowercase
  const searchValue = `%${query.trim().toLowerCase()}%`;

  // DB: search users and posts; left join to include users without posts
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

      // GROUP: aggregate by username; keep post previews as provided
      const groupedResults = results.reduce((acc, post) => {
        const username = post.username;
        const hasPost = post.post_id !== null;
        const existingUser = acc.find((user) => user.username === username);

        if (existingUser) {
          if (hasPost) {
            existingUser.posts.push({
              post_id: post.post_id,
              title: post.title,
              content_preview: post.content_preview,
              photo_url: post.photo_url || "",
            });
          }
        } else {
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
              : undefined,
          });
        }
        return acc;
      }, []);

      // RESPONSE: remove empty posts arrays for users without posts
      res.json({
        results: groupedResults.map((user) => {
          if (!user.posts) {
            delete user.posts;
          }
          return user;
        }),
      });
    }
  );
});


// ======================================================
// Profile: get own profile (auth required) with post counts
// SECURITY: path userId must equal token userId
// ======================================================
app.get("/api/users/:userId/profile", verifyToken, (req, res) => {
  const userId = req.params.userId;

  // AUTHZ: enforce self-only access
  if (req.userId.toString() !== userId) {
    return res
      .status(403)
      .json({ error: "You are not authorized to view this profile" });
  }

  // DB: profile with follower/following/post counts
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

  // DB: list posts for the user (active only)
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

    pool.query(postSql, [userId], (postError, postResults) => {
      if (postError) {
        return res
          .status(500)
          .json({ error: "Database error while fetching user posts" });
      }

      // RESPONSE: keep data shape as-is
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

      res.json(response);
    });
  });
});


// ======================================================
// Profile: public view of any user's profile (auth required)
// NOTE: includes post list with parsed media arrays
// ======================================================
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

      // MEDIA: parse photo_url/video_url if they are JSON strings; keep arrays as-is
      const formattedPosts = postResults.map((post) => {
        let photos = [];
        let videos = [];

        if (Array.isArray(post.photo_url)) {
          photos = post.photo_url;
        } else if (typeof post.photo_url === "string") {
          try {
            photos = JSON.parse(post.photo_url);
          } catch (e) {
            console.error("Error parsing photo_url:", e.message);
          }
        }

        if (Array.isArray(post.video_url)) {
          videos = post.video_url;
        } else if (typeof post.video_url === "string") {
          try {
            videos = JSON.parse(post.video_url);
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
          photos,
          videos,
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


// ======================================================
// Profile: update own profile (with optional image upload)
// SECURITY: verifyToken; username uniqueness check
// ======================================================
app.put("/api/users/:userId/profile", verifyToken, upload.single("profileImage"), (req, res) => {
    const userId = req.params.userId;

    // INPUT: collect fields; picture optional
    let { username, bio, gender, birthday } = req.body;
    const profileImage = req.file ? `/uploads/${req.file.filename}` : null;

    // VALIDATION: all required fields must be present
    if (!username || !bio || !gender || !birthday) {
      return res
        .status(400)
        .json({ error: "All fields are required: username, bio, gender, and birthday" });
    }

    // VALIDATION: birthday must be a valid date
    if (isNaN(Date.parse(birthday))) {
      return res.status(400).json({ error: "Invalid birthday format" });
    }

    // FORMAT: convert to yyyy-MM-dd (uses provided helper)
    birthday = formatDateForSQL(birthday);

    // UNIQUE: ensure username not taken by another user
    const checkUsernameSql = `SELECT id FROM users WHERE username = ? AND id != ?`;

    pool.query(checkUsernameSql, [username, userId], (checkError, checkResults) => {
      if (checkError) {
        console.error("Error checking username:", checkError);
        return res.status(500).json({ error: "Database error while checking username" });
      }

      if (checkResults.length > 0) {
        return res.status(400).json({ error: "Username is already in use" });
      }

      // DB: build update query; include picture if uploaded
      let updateProfileSql = `UPDATE users SET username = ?, bio = ?, gender = ?, birthday = ?`;
      const updateData = [username, bio, gender, birthday];

      if (profileImage) {
        updateProfileSql += `, picture = ?`;
        updateData.push(profileImage);
      }

      updateProfileSql += ` WHERE id = ?;`;
      updateData.push(userId);

      pool.query(updateProfileSql, updateData, (error, results) => {
        if (error) {
          console.error("Error updating profile:", error);
          return res.status(500).json({ error: "Database error while updating user profile" });
        }

        if (results.affectedRows === 0) {
          return res.status(404).json({ error: "User not found" });
        }

        res.json({
          message: "Profile updated successfully",
          profileImage: profileImage || "No image uploaded",
        });
      });
    });
  }
);


// Format birthday to SQL (YYYY-MM-DD)
function formatDateForSQL(dateString) {
  const dateObj = new Date(dateString);
  const year = dateObj.getFullYear();
  const month = String(dateObj.getMonth() + 1).padStart(2, '0');
  const day = String(dateObj.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

// Follow / Unfollow user
app.post("/api/users/:userId/follow/:followingId", verifyToken, (req, res) => {
  const { userId, followingId } = req.params;

  if (req.userId.toString() !== userId) {
    return res.status(403).json({ error: "Unauthorized" });
  }

  pool.query("SELECT 1 FROM users WHERE id = ?", [followingId], (error, userCheck) => {
    if (error) return res.status(500).json({ error: "DB error" });
    if (userCheck.length === 0) return res.status(404).json({ error: "User not found" });

    pool.query(
      "SELECT 1 FROM follower_following WHERE follower_id = ? AND following_id = ?",
      [userId, followingId],
      (error, followCheck) => {
        if (error) return res.status(500).json({ error: "DB error" });

        if (followCheck.length > 0) {
          pool.query(
            "DELETE FROM follower_following WHERE follower_id = ? AND following_id = ?",
            [userId, followingId],
            (error) => {
              if (error) return res.status(500).json({ error: "DB error" });
              return res.status(200).json({ message: "Unfollowed" });
            }
          );
        } else {
          pool.query(
            "INSERT INTO follower_following (follower_id, following_id) VALUES (?, ?)",
            [userId, followingId],
            (error) => {
              if (error) return res.status(500).json({ error: "DB error" });
              return res.status(201).json({ message: "Followed" });
            }
          );
        }
      }
    );
  });
});

// Check follow status
app.get("/api/users/:userId/follow/:followingId/status", verifyToken, (req, res) => {
  const { userId, followingId } = req.params;

  if (req.userId.toString() !== userId) {
    return res.status(403).json({ error: "Unauthorized" });
  }

  pool.query("SELECT 1 FROM users WHERE id = ?", [followingId], (error, userCheck) => {
    if (error) return res.status(500).json({ error: "DB error" });
    if (userCheck.length === 0) return res.status(404).json({ error: "User not found" });

    pool.query(
      "SELECT 1 FROM follower_following WHERE follower_id = ? AND following_id = ?",
      [userId, followingId],
      (error, followCheck) => {
        if (error) return res.status(500).json({ error: "DB error" });
        return res.status(200).json({ isFollowing: followCheck.length > 0 });
      }
    );
  });
});

// Add comment
app.post("/api/posts/:postId/comment", verifyToken, (req, res) => {
  try {
    const { postId } = req.params;
    const { content } = req.body;
    const userId = req.userId;

    if (!content || content.trim() === "") {
      return res.status(400).json({ error: "Content cannot be empty" });
    }

    pool.query(
      "INSERT INTO comments (post_id, user_id, comment_text) VALUES (?, ?, ?)",
      [postId, userId, content],
      (error, results) => {
        if (error) return res.status(500).json({ error: "DB error" });
        res.status(201).json({
          message: "Comment added",
          comment_id: results.insertId,
          post_id: postId,
          user_id: userId,
          content,
        });
      }
    );
  } catch {
    res.status(500).json({ error: "Internal server error" });
  }
});

// Delete comment
app.delete("/api/posts/:postId/comment/:commentId", verifyToken, (req, res) => {
  const { postId, commentId } = req.params;
  const userId = req.userId;

  pool.query(
    "SELECT 1 FROM comments WHERE id = ? AND user_id = ? AND post_id = ?",
    [commentId, userId, postId],
    (err, results) => {
      if (err) return res.status(500).json({ error: "DB error" });
      if (results.length === 0) {
        return res.status(404).json({ error: "Comment not found or unauthorized" });
      }

      pool.query(
        "DELETE FROM comments WHERE id = ? AND user_id = ? AND post_id = ?",
        [commentId, userId, postId],
        (err, delResults) => {
          if (err) return res.status(500).json({ error: "DB error" });
          if (delResults.affectedRows === 0) {
            return res.status(404).json({ error: "Comment not deleted" });
          }

          pool.query(
            "DELETE FROM notifications WHERE comment_id = ?",
            [commentId],
            (err) => {
              if (err) return res.status(500).json({ error: "DB error" });
              return res.status(200).json({ message: "Comment & notification deleted" });
            }
          );
        }
      );
    }
  );
});


//########################################################   Notification API  #######################################################



// ======================================================
// Notifications: list for current user (post actions + ads status)
// SECURITY: verifyToken; filters by receiver (post owner) or ad owner
// ======================================================
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
    n.ads_id,
    s.username AS sender_name,
    s.picture AS sender_picture, 
    p_owner.username AS receiver_name,
    c.comment_text AS comment_content  
  FROM notifications n
  LEFT JOIN users s ON n.user_id = s.id
  LEFT JOIN posts p ON n.post_id = p.id
  LEFT JOIN users p_owner ON p.user_id = p_owner.id
  LEFT JOIN comments c ON c.id = n.comment_id
  WHERE n.action_type IN ('comment', 'like', 'follow', 'bookmark', 'ads_status_change')
    AND (
      (n.action_type = 'ads_status_change' AND n.user_id = ?)
      OR
      (n.action_type IN ('comment', 'like', 'follow', 'bookmark') AND p_owner.id = ?)
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



// ======================================================
// Notifications: create (comment always new; like/follow idempotent)
// SECURITY: verifyToken required
// ======================================================
app.post("/api/notifications", verifyToken, (req, res) => {
  const { user_id, post_id, action_type, content, comment_id } = req.body;

  if (!user_id || !action_type) {
    return res.status(400).json({ error: "Missing required fields: user_id or action_type" });
  }

  if (action_type === 'comment') {
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
      return res.status(201).json({
        message: "Notification created successfully",
        notification_id: results.insertId,
      });
    });
  } else {
    // ⬇️ เพิ่ม bookmark เข้าไปในลิสต์เดียวกับ like/follow
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

      if (checkResults.length > 0) {
        const existingNotificationId = checkResults[0].id;

        // ⬇️ ลบเมื่อ action ซ้ำสำหรับ like/follow/bookmark
        if (action_type === 'like' || action_type === 'follow' || action_type === 'bookmark') {
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
          return res.status(201).json({
            message: "Notification created successfully",
            notification_id: results.insertId,
          });
        });
      }
    });
  }
});


// ======================================================
// Notifications: mark one as read (only post owner)
// SECURITY: verifyToken; join to confirm ownership
// ======================================================
app.put("/api/notifications/:id/read", verifyToken, (req, res) => {
  const { id } = req.params;
  const userId = req.userId;

  console.log("Notification ID:", id);
  console.log("User ID from Token (Post Owner):", userId);

  const updateReadStatusSql = `
    UPDATE notifications n
    JOIN posts p ON n.post_id = p.id
    SET n.read_status = 1
    WHERE n.id = ? AND p.user_id = ?;
  `;

  pool.query(updateReadStatusSql, [id, userId], (error, results) => {
    if (error) {
      console.error("Database error during updating read status:", error);
      return res.status(500).json({ error: "Error updating read status" });
    }
    if (results.affectedRows === 0) {
      console.warn(`Notification not found or you are not the owner of the post (User ID: ${userId})`);
      return res.status(404).json({ message: "Notification not found or you are not the owner of the post" });
    }

    console.log("Notification marked as read for ID:", id);
    res.json({ message: "Notification marked as read" });
  });
});


// ======================================================
// Notifications: delete by (user_id, post_id, action_type)
// SECURITY: verifyToken; body must specify all keys
// ======================================================
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
  });
});


// ======================================================
// Ads: notify status change (admin triggers)
// SECURITY: verifyToken; creates ads_status_change notification
// ======================================================
app.post("/api/ads/:id/notify-status-change", verifyToken, (req, res) => {
  const adId = req.params.id;
  const { new_status, admin_notes } = req.body;

  const getAdSql = `SELECT * FROM ads WHERE id = ?`;
  pool.query(getAdSql, [adId], (adErr, adResults) => {
    if (adErr || adResults.length === 0) {
      return res.status(404).json({ error: "Ad not found" });
    }
    const ad = adResults[0];

    let content = `Your Ad (${ad.title}) Has change to "${new_status}"`;
    if (new_status === "rejected" && admin_notes) {
      content += `\nReason: ${admin_notes}`;
    }

    const insertNotificationSql = `
      INSERT INTO notifications (user_id, action_type, content)
      VALUES (?, ?, ?)
    `;
    pool.query(
      insertNotificationSql,
      [ad.user_id, "ads_status_change", content],
      (notiErr) => {
        if (notiErr) {
          return res.status(500).json({ error: "Failed to save notification" });
        }
        res.status(201).json({ message: "Ad status notification sent successfully" });
      }
    );
  });
});


// ======================================================
// Orders/Ads: helper to update order/ad and send notifications
// NOTE: transactional flow uses provided connection; keeps existing logic
// ======================================================
function proceedUpdateOrderAndAd(connection, orderId, slipImagePath, renewAdsId, packageId, originalAdExpirationDate, originalAdShowAt, canUpload, res) {
  if (!canUpload) {
      return connection.rollback(() => {
          connection.release();
          res.status(400).json({ error: 'Cannot upload slip due to invalid status' });
      });
  }

  const updateOrderSql = 'UPDATE orders SET slip_image = ?, status = "paid", updated_at = NOW() WHERE id = ?';
  connection.query(updateOrderSql, [slipImagePath, orderId], (updateOrderErr) => {
      if (updateOrderErr) {
          return connection.rollback(() => {
              connection.release();
              console.error(`[ERROR] Database error updating slip_image for order ${orderId}:`, updateOrderErr);
              res.status(500).json({ error: 'Error saving slip' });
          });
      }

      if (renewAdsId !== null) {
          const getDurationSql = 'SELECT duration_days FROM ad_packages WHERE package_id = ?';
          connection.query(getDurationSql, [packageId], (durationErr, durationResults) => {
              if (durationErr || durationResults.length === 0) {
                  console.error(`[ERROR] Failed to get duration for package ${packageId} on order ${orderId}:`, durationErr || 'No package info found');
                  return connection.commit(() => {
                      connection.release();
                      res.json({ message: 'Slip uploaded successfully, but there was a problem renewing the ad. Please contact admin', slip_path: slipImagePath });
                  });
              }
              const duration_days = durationResults[0].duration_days;

              // EXPIRY: add duration to existing expiration_date
              const newExpirationDate = new Date(originalAdExpirationDate);
              newExpirationDate.setDate(newExpirationDate.getDate() + duration_days);

              const updateAdsSql = `
                  UPDATE ads
                  SET status = 'active',
                      expiration_date = ?,
                      updated_at = NOW()
                  WHERE id = ?;
              `;
              connection.query(updateAdsSql, [newExpirationDate, renewAdsId], (updateAdsErr) => {
                  if (updateAdsErr) {
                      console.error(`[ERROR] Database error updating ad ${renewAdsId} for order ${orderId}:`, updateAdsErr);
                      return connection.commit(() => {
                          connection.release();
                          res.json({ message: 'Slip uploaded successfully, but there was a problem renewing the ad. Please contact admin', slip_path: slipImagePath });
                      });
                  }
                  console.log(`[INFO] Ad ${renewAdsId} successfully renewed and set to 'active' via order ${orderId}.`);
                  console.log(`[INFO] Your ad has been renewed ${duration_days} days!`);

                  // NOTIFY: send renewal success message
                  const notiMsg = `Your ad has been renewed ${duration_days} days!.`;
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
                                      res.json({ message: `Slip uploaded successfully! Your ad has been renewed for ${duration_days} days.`, slip_path: slipImagePath });
                                  });
                              }
                          );
                      } else {
                          connection.commit(() => {
                              connection.release();
                              res.json({ message: `Slip uploaded successfully! Your ad has been renewed for ${duration_days} days.`, slip_path: slipImagePath });
                          });
                        }
                    }
                  );
              });
          });

      } else {
          // NEW AD: mark as paid and notify
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
                      res.json({ message: 'Slip uploaded successfully! Please wait for admin to review the slip', slip_path: slipImagePath });
                  });
              });
          });
      }
  });
}


// ======================================================
// Locale: format date to Thai with Buddhist calendar (utility)
// ======================================================
function formatThaiDate(dateString) {
  const date = new Date(dateString);
  const formatter = new Intl.DateTimeFormat('th-TH', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      calendar: 'buddhist'
  });
  return formatter.format(date);
}


// ======================================================
// Ads: generic notify helper for status changes
// NOTE: special case when status becomes 'active' after paid renewal
// ======================================================
function notifyAdsStatusChange(adId, newStatus, adminNotes = null, callback) {
  pool.query('SELECT user_id, expiration_date FROM ads WHERE id = ?', [adId], (err, adsResults) => {
      if (err || adsResults.length === 0) {
          return callback(err || new Error('Ad not found'));
      }
      const { user_id, expiration_date } = adsResults[0];
      let content = '';

      pool.query(
          `SELECT package_id FROM orders WHERE renew_ads_id = ? AND status = 'paid'`,
          [adId],
          (err, orderResults) => {
              if (err) {
                  return callback(err);
              }

              if (newStatus === 'active' && orderResults.length > 0) {
                  const renewedPackageId = orderResults[0].package_id;

                  pool.query(
                      `SELECT duration_days FROM ad_packages WHERE package_id = ?`,
                      [renewedPackageId],
                      (err, packageResults) => {
                          if (err) {
                              return callback(err);
                          }

                          let renewedDays = 'ไม่ระบุ';
                          if (packageResults.length > 0) {
                              renewedDays = packageResults[0].duration_days;
                          }

                          const formattedExpirationDate = formatThaiDate(expiration_date);
                          content = `Your ad has been successfully renewed for ${renewedDays} days. The new expiration date is ${formattedExpirationDate}.`;

                          pool.query(
                              `INSERT INTO notifications (user_id, action_type, content, ads_id) VALUES (?, 'ads_status_change', ?, ?)`,
                              [user_id, content, adId],
                              callback
                          );
                      }
                  );
              } else {
                  switch (newStatus) {
                      case 'approved':
                          content = 'Your ad has been reviewed. Please transfer payment to display it.';
                          break;
                      case 'active':
                          content = 'Your ad has been approved for display.';
                          break;
                      case 'rejected':
                          content = `Your ad was rejected. Reason: ${adminNotes || '-'}`;
                          break;
                      case 'paid':
                          content = 'Your ad payment has been completed. Waiting for admin review.';
                          break;
                      case 'expired':
                          content = 'Your ad has expired.';
                          break;
                      case 'expiring_soon':
                          content = 'Your ad will expire in 3 days. Please renew to ensure continuous display.';
                          break;
                      default:
                          content = `Your ad status has changed to ${newStatus}`;
                  }

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


// ======================================================
// Cron helper: check ads expiring in 3 days and notify
// NOTE: call from your scheduler; does not run automatically
// ======================================================
function checkExpiringAds() {
    console.log('[INFO] Checking for expiring ads...');
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



// ######################################################  Bookmark API  ######################################################

// ======================================================
// Bookmark: toggle (สร้าง/ลบ Notification + log interaction)
// SECURITY: verifyToken
// ======================================================
app.post("/api/posts/:postId/bookmark", verifyToken, (req, res) => {
  const { postId } = req.params;
  const userId = req.userId;

  if (!postId) return res.status(400).json({ error: "Post ID required" });

  pool.query(
    "SELECT 1 FROM bookmarks WHERE user_id = ? AND post_id = ?",
    [userId, postId],
    (err, results) => {
      if (err) {
        console.error("Database error during checking bookmark status:", err);
        return res.status(500).json({ error: "Error checking bookmark status" });
      }

      if (results.length > 0) {
        // ========= UNBOOKMARK =========
        pool.query(
          "DELETE FROM bookmarks WHERE user_id = ? AND post_id = ?",
          [userId, postId],
          (delErr) => {
            if (delErr) {
              console.error("Database error during removing bookmark:", delErr);
              return res.status(500).json({ error: "Error removing bookmark" });
            }

            // ลบ noti ของ bookmark
            const delNotiSql = `
              DELETE FROM notifications
              WHERE user_id = ? AND post_id = ? AND action_type = 'bookmark'
            `;
            pool.query(delNotiSql, [userId, postId], (nErr) => {
              if (nErr) console.error("Delete bookmark noti error:", nErr);
            });

            // log interaction: unbookmark
            const logUnbookmarkSql = `
              INSERT INTO user_interactions (user_id, post_id, action_type, content)
              VALUES (?, ?, 'unbookmark', NULL)
            `;
            pool.query(logUnbookmarkSql, [userId, postId], (iErr) => {
              if (iErr) console.error("Interaction log error (unbookmark):", iErr);
            });

            return res.status(200).json({ message: "Bookmark removed" });
          }
        );
      } else {
        // ========= BOOKMARK =========
        pool.query(
          "INSERT INTO bookmarks (user_id, post_id) VALUES (?, ?)",
          [userId, postId],
          (insErr) => {
            if (insErr) {
              console.error("Database error during adding bookmark:", insErr);
              return res.status(500).json({ error: "Error adding bookmark" });
            }

            // noti: สร้าง content แบบเดียวกับ like/comment
            const contentMsg = `User ${userId} performed action: bookmark on post ${postId}`;

            // กัน noti ซ้ำ
            const checkNotiSql = `
              SELECT id FROM notifications
              WHERE user_id = ? AND post_id = ? AND action_type = 'bookmark'
              LIMIT 1
            `;
            pool.query(checkNotiSql, [userId, postId], (cErr, cRows) => {
              if (cErr) {
                console.error("Check bookmark noti error:", cErr);
              } else if (cRows.length === 0) {
                const addNotiSql = `
                  INSERT INTO notifications (user_id, post_id, action_type, content)
                  VALUES (?, ?, 'bookmark', ?)
                `;
                pool.query(addNotiSql, [userId, postId, contentMsg], (nErr) => {
                  if (nErr) console.error("Insert bookmark noti error:", nErr);
                });
              }
            });

            // log interaction: bookmark
            const logBookmarkSql = `
              INSERT INTO user_interactions (user_id, post_id, action_type, content)
              VALUES (?, ?, 'bookmark', NULL)
            `;
            pool.query(logBookmarkSql, [userId, postId], (iErr) => {
              if (iErr) console.error("Interaction log error (bookmark):", iErr);
            });

            return res.status(201).json({ message: "Bookmarked" });
          }
        );
      }
    }
  );
});


// ======================================================
// Bookmark: list detailed bookmarks with counts/follow flag
// SECURITY: verifyToken
// NOTE: duplicates with another /api/bookmarks GET below (kept as-is)
// ======================================================
app.get("/api/bookmarks", verifyToken, (req, res) => {
  const user_id = req.userId;

  const sql = `
    SELECT p.id AS post_id, p.title, p.content, p.photo_url, p.video_url, p.updated_at,
      (SELECT COUNT(*) FROM likes WHERE post_id = p.id) AS like_count,
      (SELECT COUNT(*) FROM comments WHERE post_id = p.id) AS comment_count,
      u.id AS user_id, u.username AS author_username, u.picture AS author_profile_image,
      CASE WHEN (SELECT COUNT(*) FROM follower_following WHERE follower_id = ? AND following_id = u.id) > 0 THEN TRUE ELSE FALSE END AS is_following
    FROM bookmarks b
    JOIN posts p ON b.post_id = p.id
    JOIN users u ON p.user_id = u.id
    WHERE b.user_id = ? AND p.status = 'active'
    ORDER BY b.created_at DESC;
  `;

  pool.query(sql, [user_id, user_id], (err, results) => {
    if (err) return res.status(500).json({ error: "DB error" });
    if (results.length === 0) return res.status(404).json({ message: "No bookmarks" });

    // SHAPE: normalize media to arrays
    const bookmarks = results.map((post) => ({
      post_id: post.post_id,
      title: post.title,
      content: post.content,
      created_at: post.updated_at,
      like_count: post.like_count,
      comment_count: post.comment_count,
      photos: parseJsonSafe(post.photo_url),
      videos: parseJsonSafe(post.video_url),
      author: {
        user_id: post.user_id,
        username: post.author_username,
        profile_image: post.author_profile_image,
      },
      is_following: !!post.is_following,
    }));

    res.json({ bookmarks });
  });
});


// ======================================================
// Util: safe JSON parse returning array
// ======================================================
function parseJsonSafe(data) {
  if (typeof data === "string") {
    try {
      return JSON.parse(data);
    } catch {
      return [];
    }
  }
  return Array.isArray(data) ? data : [];
}


// ======================================================
// Bookmark: add (explicit)  — ใส่ content ตอนสร้าง noti
// SECURITY: verifyToken
// ======================================================
app.post("/api/bookmarks", verifyToken, (req, res) => {
  const { post_id } = req.body;
  const user_id = req.userId;

  if (!post_id) {
    return res.status(400).json({ error: "Post ID is required" });
  }

  pool.query("INSERT INTO bookmarks (user_id, post_id) VALUES (?, ?)", [user_id, post_id], (err) => {
    if (err) {
      console.error("Database error during adding bookmark:", err);
      return res.status(500).json({ error: "Error adding bookmark" });
    }

    // noti + content
    const contentMsg = `User ${user_id} performed action: bookmark on post ${post_id}`;
    const addNotiSql = `
      INSERT INTO notifications (user_id, post_id, action_type, content)
      VALUES (?, ?, 'bookmark', ?)
    `;
    pool.query(addNotiSql, [user_id, post_id, contentMsg], (nErr) => {
      if (nErr) console.error("Database error during bookmark notification:", nErr);
    });

    // interaction
    const addInteractionSql = `
      INSERT INTO user_interactions (user_id, post_id, action_type, content)
      VALUES (?, ?, 'bookmark', NULL)
    `;
    pool.query(addInteractionSql, [user_id, post_id], (iErr) => {
      if (iErr) console.error("Interaction log error (bookmark):", iErr);
    });

    res.status(201).json({ message: "Post bookmarked successfully" });
  });
});



// ======================================================
// Bookmark: delete (explicit)
// SECURITY: verifyToken
// ======================================================
app.delete("/api/bookmarks", verifyToken, (req, res) => {
  const { post_id } = req.body;
  const user_id = req.userId;

  if (!post_id) {
    return res.status(400).json({ error: "Post ID is required" });
  }

  const deleteBookmarkSql = "DELETE FROM bookmarks WHERE user_id = ? AND post_id = ?";
  pool.query(deleteBookmarkSql, [user_id, post_id], (err, results) => {
    if (err) {
      console.error("Database error during deleting bookmark:", err);
      return res.status(500).json({ error: "Error deleting bookmark" });
    }
    if (results.affectedRows === 0) {
      return res.status(404).json({ message: "Bookmark not found or you are not authorized to delete" });
    }

    // --- ลบ Notifications ของ bookmark (เหมือนเดิม) ---
    const delNotiSql = `
      DELETE FROM notifications
      WHERE user_id = ? AND post_id = ? AND action_type = 'bookmark'
    `;
    pool.query(delNotiSql, [user_id, post_id], (nErr) => {
      if (nErr) console.error("Database error during deleting bookmark notification:", nErr);
    });

    // --- user_interactions: unbookmark (fire-and-forget) ---
    const addInteractionSql = `
      INSERT INTO user_interactions (user_id, post_id, action_type, content)
      VALUES (?, ?, 'unbookmark', NULL)
    `;
    pool.query(addInteractionSql, [user_id, post_id], (iErr) => {
      if (iErr) console.error("Interaction log error (unbookmark):", iErr);
    });

    return res.json({ message: "Bookmark deleted successfully" });
  });
});




// ======================================================
// Bookmark: list simple (duplicate endpoint kept as-is)
// SECURITY: verifyToken
// ======================================================
app.get("/api/bookmarks", verifyToken, (req, res) => {
  const user_id = req.userId;

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


// ======================================================
// Bookmark: check bookmark status for a post
// SECURITY: verifyToken
// ======================================================
app.get("/api/posts/:postId/bookmark/status", verifyToken, (req, res) => {
  const { postId } = req.params;
  const userId = req.userId;

  const checkBookmarkSql = "SELECT * FROM bookmarks WHERE user_id = ? AND post_id = ?";

  pool.query(checkBookmarkSql, [userId, postId], (err, results) => {
    if (err) {
      console.error("Database error during checking bookmark status:", err);
      return res.status(500).json({ error: "Error checking bookmark status" });
    }

    const isBookmarked = results.length > 0;
    res.status(200).json({ isBookmarked });
  });
});


// ======================================================
// Feed: posts from followed users
// SECURITY: verifyToken
// ======================================================
app.get("/api/following/posts", verifyToken, (req, res) => {
  const userId = req.userId;

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

    if (results.length === 0) {
      return res.status(200).json({ message: "No posts from followed users.", posts: [] });
    }

    // SHAPE: keep arrays only if already arrays (no JSON parse here)
    const parsedResults = results.map((post) => {
      const photoUrls = Array.isArray(post.photoUrl) ? post.photoUrl : [];
      const videoUrls = Array.isArray(post.videoUrl) ? post.videoUrl : [];
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
        isLiked: !!post.is_liked,
      };
    });

    res.status(200).json({ posts: parsedResults });
  });
});



//########################################################   Report API  ########################################################


// ======================================================
// Report: create a report for a post (self)
// SECURITY: requires verifyToken
// ======================================================
app.post("/api/posts/:postId/report", verifyToken, (req, res) => {
  const { postId } = req.params;
  const { reason } = req.body;
  const userId = req.userId; // from token

  // VALIDATION
  if (!reason || reason.trim() === "") {
    return res.status(400).json({ error: "Report reason is required" });
  }

  // DB: insert report
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


// ======================================================
// Users: soft delete user; hard delete their posts; delete follow links
// SECURITY: only the user themselves or admin
// NOTE: multi-step deletes; errors are handled per step
// ======================================================
app.delete("/api/users/:id", verifyToken, (req, res) => {
  const { id } = req.params;
  const user_id = req.userId; // from token

  // AUTHZ: self or admin
  if (parseInt(user_id) !== parseInt(id) && req.role !== "admin") {
    return res.status(403).json({ error: "You do not have permission to delete this user." });
  }

  // DB: delete posts (hard delete)
  const deletePostsSql = "DELETE FROM posts WHERE user_id = ?";
  pool.query(deletePostsSql, [id], (postErr, postResults) => {
    if (postErr) {
      console.error("Database error during post deletion:", postErr);
      return res.status(500).json({ error: "Database error during post deletion" });
    }

    // DB: delete follow relations (both directions)
    const deleteFollowsSql = "DELETE FROM follower_following WHERE follower_id = ? OR following_id = ?";
    pool.query(deleteFollowsSql, [id, id], (followErr, followResults) => {
      if (followErr) {
        console.error("Database error during follow deletion:", followErr);
        return res.status(500).json({ error: "Database error during follow deletion" });
      }

      // DB: soft delete user
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
          deletedPostsCount: postResults.affectedRows,
          deletedFollowsCount: followResults.affectedRows
        });
      });
    });
  });
});


//########################################################   Follow API  ########################################################


// ======================================================
// Follow: list users I follow (public endpoint as provided)
// NOTE: no auth here by design in original code
// ======================================================
app.get("/api/users/following/:userId", (req, res) => {
  const { userId } = req.params;

  // VALIDATION
  if (!userId) {
    return res.status(400).json({ error: "User ID not provided" });
  }

  // DB: fetch following
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

    if (results.length === 0) {
      return res.status(404).json({ message: "No following found" });
    }

    res.json(results);
  });
});


// ======================================================
// Follow: list my followers (public endpoint as provided)
// NOTE: no auth here by design in original code
// ======================================================
app.get("/api/users/followers/:userId", (req, res) => {
  const { userId } = req.params;

  // VALIDATION
  if (!userId) {
    return res.status(400).json({ error: "User ID not provided" });
  }

  // DB: fetch followers
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

    if (results.length === 0) {
      return res.status(404).json({ message: "No followers found" });
    }

    res.json(results);
  });
});


// ======================================================
// Follow: search within my following (auth required)
// SECURITY: verifyToken; LIKE on username (active only)
// ======================================================
app.get("/api/users/search/following", verifyToken, (req, res) => {
  const { query } = req.query;
  const followerId = req.userId; // from token

  // VALIDATION
  if (!query || query.trim() === "") {
    return res.status(400).json({ error: "Search query is required" });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`;

  // DB: search following by username
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

    res.status(200).json(results || []);
  });
});


// ======================================================
// Follow: search within my followers (auth required)
// SECURITY: verifyToken; LIKE on username (active only)
// ======================================================
app.get("/api/users/search/followers", verifyToken, (req, res) => {
  const { query } = req.query;
  const followingId = req.userId; // from token (the one being followed)

  // VALIDATION
  if (!query || query.trim() === "") {
    return res.status(400).json({ error: "Search query is required" });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`;

  // DB: search followers by username
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

    res.status(200).json(results || []);
  });
});


// ======================================================
// Bookmarks: check bookmark status for a post (auth required)
// SECURITY: verifyToken
// ======================================================
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

/* ================================
   Admin: Login (token 1 ชั่วโมง)
   SECURITY: ตรวจ role=admin, active เท่านั้น
================================ */
app.post("/api/admin/login", async (req, res) => {
  try {
      const { email, password } = req.body;
      const ipAddress = req.headers["x-forwarded-for"] || req.socket.remoteAddress;

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

                  const remainingAttempts = Math.max(0, 5 - (user.failed_attempts + 1));
                  let message = `Email or Password is incorrect.`;
                  if (remainingAttempts > 0) {
                      message += ` You have ${remainingAttempts} attempts left.`;
                  } else {
                      message += ` Your account might be locked.`;
                  }
                  return res.status(401).json({ message });
              }

              // reset failed attempts + issue JWT (1h)
              const resetFailSql =
                  "UPDATE users SET failed_attempts = 0, last_login = NOW(), last_login_ip = ? WHERE id = ?";
              pool.query(resetFailSql, [ipAddress, user.id], (err) => {
                  if (err) {
                      console.error("Error resetting failed attempts or updating login time:", err);
                      return res.status(500).json({ error: "Error updating login details." });
                  }

                  const token = jwt.sign({ id: user.id, role: user.role }, JWT_SECRET, { expiresIn: '1h' });

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


/* ================================
   Admin: Dashboard snapshot (รวมหลายกราฟ)
   SECURITY: authenticateToken + authorizeAdmin
================================ */
app.get("/api/admin/dashboard", authenticateToken, authorizeAdmin, (req, res) => {
    const newUsersQuery = "SELECT DATE_FORMAT(created_at, '%Y-%m') AS month_year, COUNT(*) AS new_users FROM users WHERE role = 'user' GROUP BY month_year ORDER BY month_year DESC;";
    const totalPostsQuery = "SELECT DATE_FORMAT(updated_at, '%Y-%m') AS month_year, COUNT(*) AS total_posts FROM posts GROUP BY month_year ORDER BY month_year DESC;";
    const categoryPopularityQuery = "SELECT CASE WHEN Electronics_Gadgets = 1 THEN 'Electronics & Gadgets' WHEN Furniture = 1 THEN 'Furniture' WHEN Outdoor_Gear = 1 THEN 'Outdoor Gear' WHEN Beauty_Products = 1 THEN 'Beauty Products' WHEN Accessories = 1 THEN 'Accessories' ELSE 'Other' END AS CategoryName, SUM(PostEngagement) AS TotalEngagement, SUM(CASE WHEN Male = 1 THEN PostEngagement ELSE 0 END) AS MaleEngagement, SUM(CASE WHEN Female = 1 THEN PostEngagement ELSE 0 END) AS FemaleEngagement, SUM(CASE WHEN Male = 0 AND Female = 0 THEN PostEngagement ELSE 0 END) AS OtherEngagement FROM contentbasedview GROUP BY CategoryName ORDER BY TotalEngagement DESC;";
    const ageInterestQuery = "SELECT CASE WHEN Age BETWEEN 18 AND 25 THEN '18-25' WHEN Age BETWEEN 26 AND 35 THEN '26-35' WHEN Age > 35 THEN '36+' ELSE 'Other' END AS AgeGroup, CASE WHEN Electronics_Gadgets = 1 THEN 'Electronics & Gadgets' WHEN Furniture = 1 THEN 'Furniture' WHEN Outdoor_Gear = 1 THEN 'Outdoor Gear' WHEN Beauty_Products = 1 THEN 'Beauty Products' WHEN Accessories = 1 THEN 'Accessories' ELSE 'Other' END AS CategoryName, SUM(PostEngagement) AS TotalEngagement FROM contentbasedview GROUP BY AgeGroup, CategoryName ORDER BY AgeGroup, TotalEngagement DESC;";

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
  
            pool.query(categoryPopularityQuery, (categoryPopularityError, categoryPopularityResults) => {
                if (categoryPopularityError) {
                    console.error("Database error fetching category popularity:", categoryPopularityError);
                    return res.status(500).json({ error: "Error fetching category popularity data" });
                }
  
                pool.query(ageInterestQuery, (ageInterestError, ageInterestResults) => {
                    if (ageInterestError) {
                        console.error("Database error fetching age interest data:", ageInterestError);
                        return res.status(500).json({ error: "Error fetching age interest data" });
                    }
                    
                    res.json({
                        new_users_per_month: newUsersResults,
                        total_posts_per_month: totalPostsResults,
                        category_popularity: categoryPopularityResults,
                        age_interest: ageInterestResults,
                    });
                });
            });
        });
    });
  });


/* ================================
   Admin: รายงานโพสต์ pending (ซ้ำกับ /api/reports ด้านบนของมึง)
   SECURITY: verifyToken + role=admin
   NOTE: endpoint เดิมซ้ำ มีไว้ตามเดิม
================================ */
app.get("/api/reports", verifyToken, (req, res) => {
  const role = req.role;
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


/* ================================
   Ads: random 1 ตัว (active) ถ่วงน้ำหนัก display_count ต่ำก่อน
   NOTE: คืนเป็น array ตามของเดิม
================================ */
app.get('/api/ads/random', (req, res) => {
  const sql = `
    SELECT *
    FROM ads
    WHERE status = 'active'
    ORDER BY display_count ASC, RAND()
    LIMIT 1
  `;
  pool.query(sql, (err, rows) => {
    if (err) {
      console.error('Database error during fetching random ad:', err);
      return res.status(500).json({ error: 'Error fetching random ad' });
    }
    res.json(rows);
  });
});


/* ================================
   Ads: track impression (เพิ่มจำนวน + last_shown)
================================ */
app.post('/api/ads/track', (req, res) => {
  const adId = req.body.id;
  if (!adId) return res.status(400).json({ error: 'Ad ID is required' });

  const sql = `
    UPDATE ads
    SET display_count = display_count + 1,
        last_shown = NOW()
    WHERE id = ?
  `;
  pool.query(sql, [adId], (err, result) => {
    if (err) {
      console.error('Database error during ad count update:', err);
      return res.status(500).json({ error: 'Error updating ad count' });
    }
    if (result.affectedRows === 0) {
      return res.status(404).json({ error: 'Ad not found or not updated' });
    }
    res.json({ message: 'Ad count updated successfully' });
  });
});


/* ================================
   Static: serve images under /api/uploads
================================ */
app.use("/api/uploads", express.static('uploads'));


/* ================================
   Ads: create (admin only) อัปโหลดรูปได้
   SECURITY: authenticateToken + authorizeAdmin
================================ */
app.post("/api/ads", authenticateToken, authorizeAdmin, upload.single("image"), (req, res) => {
  const { title, content, link, status, expiration_date } = req.body;
  const image = req.file ? `/uploads/${req.file.filename}` : null;
  const userId = req.user.id; // จาก authenticateToken

  if (!title || !content || !link || !image || !status || !expiration_date || !userId) {
      if (req.file) {
          require('fs').unlink(req.file.path, (err) => {
              if (err) console.error("Error deleting incomplete ad image:", err);
          });
      }
      return res.status(400).json({ error: "All required fields (title, content, link, image, status, expiration_date, user_id) are required" });
  }

  const createAdSql = `INSERT INTO ads (title, content, link, image, status, expiration_date, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)`;
  pool.query(createAdSql, [title, content, link, image, status, expiration_date, userId], (err, results) => {
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


/* ================================
   Ads: update (admin only), อัปเดตเฉพาะ field ที่ส่งมา
   SECURITY: authenticateToken + authorizeAdmin
   NOTE: ถ้า rejected ต้องมี admin_notes
================================ */
app.put("/api/admin/ads/:id", authenticateToken, authorizeAdmin, upload.single('image'), (req, res) => {
  const { id } = req.params;
  const { title, content, link, status, expiration_date, admin_notes, show_at, expired_at } = req.body;
  const image = req.file ? `/uploads/${req.file.filename}` : null;

  if (status === 'rejected' && (!admin_notes || admin_notes.trim() === '')) {
    if (req.file) {
      require('fs').unlink(req.file.path, (err) => {
        if (err) console.error('Error deleting uploaded image:', err);
      });
    }
    return res.status(400).json({ error: 'Please provide a reason (admin_notes) when rejecting the ad' });
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
    // แจ้งเตือนผู้ใช้เมื่อ status เปลี่ยน
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


/* ================================
   Ads: update status (admin only) กำหนดวัน show/expire ถ้า active
   SECURITY: authenticateToken + role=admin
================================ */
app.put('/api/admin/ads/:adId/status', authenticateToken, (req, res) => {
    if (req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Forbidden: Admins only' });
    }

    const { adId } = req.params;
    const { status, admin_notes } = req.body;

    const allowedStatus = ['approved', 'rejected', 'active'];
    if (!allowedStatus.includes(status)) {
        return res.status(400).json({ error: 'Invalid status value.' });
    }
    
    if (status === 'active') {
        const getAdInfoSql = `
            SELECT p.duration_days 
            FROM ads a
            JOIN orders o ON a.order_id = o.id
            JOIN ad_packages p ON o.package_id = p.package_id
            WHERE a.id = ?
        `;
        
        pool.query(getAdInfoSql, [adId], (err, results) => {
            if (err) return res.status(500).json({ error: 'Database error fetching ad info' });
            if (results.length === 0) return res.status(404).json({ error: 'Ad not found or package info missing.' });

            const duration = results[0].duration_days;
            
            // start อีก 2 วัน, expire = start + duration
            const showAtDate = new Date();
            showAtDate.setDate(showAtDate.getDate() + 2);
            const expirationDate = new Date(showAtDate);
            expirationDate.setDate(showAtDate.getDate() + duration);

            const showAtForSql = showAtDate.toISOString().split('T')[0];
            const expirationForSql = expirationDate.toISOString().split('T')[0];
            
            const updateSql = `
                UPDATE ads 
                SET status = 'active', show_at = ?, expiration_date = ? 
                WHERE id = ?
            `;
            
            pool.query(updateSql, [showAtForSql, expirationForSql, adId], (updateErr) => {
                if (updateErr) return res.status(500).json({ error: 'Database error activating ad' });
                res.status(200).json({ message: 'Ad activated successfully with new dates.' });
            });
        });

    } else {
        let sql = 'UPDATE ads SET status = ?';
        const params = [status];

        if (status === 'rejected' && admin_notes) {
            sql += ', admin_notes = ?';
            params.push(admin_notes);
        }
        sql += ' WHERE id = ?';
        params.push(adId);

        pool.query(sql, params, (err, result) => {
            if (err) return res.status(500).json({ error: 'Database error' });
            if (result.affectedRows === 0) return res.status(404).json({ error: 'Ad not found.' });
            res.status(200).json({ message: 'Status updated successfully.' });
        });
    }
});


/* ================================
   Ads: delete (admin only) + ลบรูปไฟล์จริงถ้ามี
   SECURITY: authenticateToken + authorizeAdmin
================================ */
app.delete("/api/ads/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;

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

          // ลบไฟล์ภาพบนดิสก์ (best-effort)
          if (imagePathToDelete) {
              const fullPath = path.join(__dirname, imagePathToDelete);
              require('fs').unlink(fullPath, (unlinkErr) => {
                  if (unlinkErr) console.error("Error deleting ad image file from disk:", unlinkErr);
              });
          }

          res.json({ message: "Ad deleted successfully" });
      });
  });
});



//########################################################   Admin API  ########################################################

/* ================================
   Ads (Admin): list all ads
   SECURITY: authenticateToken + authorizeAdmin
   NOTE: status sort แบบ custom ตามเดิม
================================ */
app.get("/api/ads", authenticateToken, authorizeAdmin, (req, res) => {
  const fetchAdsSql = `
      SELECT id, user_id, order_id, title, content, link, image, status, created_at, updated_at, expiration_date, admin_notes, show_at
      FROM ads
      ORDER BY
          FIELD(status, 'pending', 'paid', 'active', 'rejected'),
          created_at ASC;
  `;
  pool.query(fetchAdsSql, (err, results) => {
    if (err) {
      console.error("Database error during fetching ads:", err);
      return res.status(500).json({ error: "Error fetching ads" });
    }
    res.json(results);
  });
});


/* ================================
   Ads (Admin): get ad by id
   SECURITY: authenticateToken + authorizeAdmin
================================ */
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


/* ================================
   Ads (Admin): get image URL by ad id
   SECURITY: authenticateToken + authorizeAdmin
   NOTE: คืน URL (ไม่สตรีมไฟล์)
================================ */
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


/* ================================
   Posts (Admin): update post status by id
   SECURITY: verifyToken + role === 'admin'
================================ */
app.put("/api/posts/:id/status", verifyToken, (req, res) => {
  const postId = req.params.id;
  const roles = req.role;

  if (roles !== "admin") {
    return res.status(403).json({ error: "You do not have permission to update status." });
  }

  const { status } = req.body;

  const query = "UPDATE posts SET status = ? WHERE id = ?";
  pool.query(query, [status, postId], (err, results) => {
    if (err) {
      console.error("Database error during post status update:", err);
      return res.status(500).json({ error: "Internal server error" });
    }

    if (results.affectedRows === 0) {
      return res.status(404).json({ error: "Post not found or status not changed." });
    }

    res.json({ message: "Post status updated successfully." });
  });
});


/* ================================
   Users (Admin): get all users
   SECURITY: authenticateToken + authorizeAdmin
================================ */
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


/* ================================
   Users (Admin): get user by id
   SECURITY: authenticateToken + authorizeAdmin
================================ */
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


/* ================================
   Users (Admin): update user status
   SECURITY: authenticateToken + authorizeAdmin
================================ */
app.put("/api/admin/users/:id/status", authenticateToken, authorizeAdmin, (req, res) => {
   const { id } = req.params;
   const { status } = req.body;

   if (!status) {
       return res.status(400).json({ error: "Status is required" });
   }

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


/* ================================
   Users (Admin): delete user (soft delete user + hard delete posts + delete follows)
   SECURITY: authenticateToken + authorizeAdmin
   NOTE: ใช้ transaction ตามเดิม
================================ */
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

      const deletePostsSql = "DELETE FROM posts WHERE user_id = ?";
      connection.query(deletePostsSql, [id], (err, postResults) => {
        if (err) {
          return connection.rollback(() => {
            connection.release();
            console.error("Error deleting posts:", err);
            res.status(500).json({ error: "Failed to delete user's posts." });
          });
        }

        const deleteFollowsSql = "DELETE FROM follower_following WHERE follower_id = ? OR following_id = ?";
        connection.query(deleteFollowsSql, [id, id], (err, followResults) => {
          if (err) {
            return connection.rollback(() => {
              connection.release();
              console.error("Error deleting follows:", err);
              res.status(500).json({ error: "Failed to delete user's follows." });
            });
          }

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


/* ================================
   Posts (Admin): get all posts
   SECURITY: authenticateToken + authorizeAdmin
   NOTE: เรียงใหม่ล่าสุดก่อนตามเดิม
================================ */
app.get("/api/admin/posts", authenticateToken, authorizeAdmin, (req, res) => {
  const fetchPostsSql = "SELECT * FROM posts ORDER BY created_at DESC";
  pool.query(fetchPostsSql, (err, results) => {
      if (err) {
          console.error("Database error during fetching posts:", err);
          return res.status(500).json({ error: "Error fetching posts" });
      }
      res.json(results);
  });
});


/* ================================
   Posts (Admin): get one post by id
   SECURITY: authenticateToken + authorizeAdmin
================================ */
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


/* ================================
   Uploads (Admin): multiple images upload
   SECURITY: authenticateToken + authorizeAdmin
   NOTE: field name = 'images', limit 20
================================ */
const uploadRoot = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadRoot)) fs.mkdirSync(uploadRoot, { recursive: true });

app.post(
  '/api/admin/uploads',
  authenticateToken,
  authorizeAdmin,
  upload.array('images', 20),
  (req, res) => {
    const files = Array.isArray(req.files) ? req.files : [];
    const paths = files.map(f => `/uploads/${f.filename}`);
    res.json({ paths });
  }
);


/* ================================
   Posts (Admin): update fields selectively
   SECURITY: authenticateToken + authorizeAdmin
   NOTE: ตรวจค่า status เฉพาะ 'active'|'deactive' ตามเดิม
================================ */
app.put("/api/admin/posts/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;
  const { Title, content, status, ProductName, photo_url } = req.body;

  const updateFields = [];
  const updateValues = [];

  if (Title !== undefined) { updateFields.push('Title = ?'); updateValues.push(Title); }
  if (content !== undefined) { updateFields.push('content = ?'); updateValues.push(content); }
  if (status !== undefined) {
    const allowed = new Set(['active','deactive']);
    if (!allowed.has(String(status).toLowerCase())) {
      return res.status(400).json({ error: "Invalid status. Use 'active' or 'deactive'." });
    }
    updateFields.push('status = ?'); updateValues.push(status);
  }
  if (ProductName !== undefined) { updateFields.push('ProductName = ?'); updateValues.push(ProductName); }

  if (photo_url !== undefined) {
    const arr = Array.isArray(photo_url) ? photo_url
              : (typeof photo_url === 'string' && photo_url ? [photo_url] : []);
    updateFields.push('photo_url = CAST(? AS JSON)'); // column เป็น JSON ตามเดิม
    updateValues.push(JSON.stringify(arr));
  }

  updateFields.push('updated_at = NOW()');

  if (updateFields.length === 1) {
    return res.status(400).json({ error: 'No meaningful fields to update besides updated_at' });
  }

  const sql = `UPDATE posts SET ${updateFields.join(', ')} WHERE id = ?`;
  updateValues.push(id);

  pool.query(sql, updateValues, (err, results) => {
    if (err) { console.error(err); return res.status(500).json({ error: "Error updating post" }); }
    if (results.affectedRows === 0) return res.status(404).json({ error: "Post not found" });
    res.json({ message: "Post updated successfully" });
  });
});


/* ================================
   Posts (Admin): delete post + related reports (transaction)
   SECURITY: authenticateToken + authorizeAdmin
================================ */
app.delete("/api/admin/posts/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;

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

          const deleteReportsSql = "DELETE FROM reports WHERE post_id = ?";
          connection.query(deleteReportsSql, [id], (err, reportsResults) => {
              if (err) {
                  return connection.rollback(() => {
                      connection.release();
                      console.error("Database error during reports deletion:", err);
                      res.status(500).json({ error: "Error deleting related reports" });
                  });
              }

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


/* ================================
   Reports (Admin): list reported posts (all statuses)
   SECURITY: authenticateToken + authorizeAdmin
   NOTE: sort ตาม status priority + reported_at DESC ตามเดิม
================================ */
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
          connection.release();
          if (error) {
              console.error("Error fetching reported posts:", error);
              return res.status(500).json({ error: "Error fetching reported posts: " + error.message });
          }
          console.log("Fetched Reported Posts:", results);
          res.json(results);
      });
  });
});



// ========================= Reports & Categories (Admin) & Orders =========================
// หมายเหตุรวม: ทุกจุดคงพฤติกรรมเดิม 100% เพิ่มเฉพาะคอมเมนต์/รูปแบบให้อ่านง่าย

/* ----------------------------------------------------------------
   ADMIN: Update report status by reportId
   - newStatus: 'pending' | 'block' | 'normally'
   - ใช้ Transaction: อัปเดตรายงานทั้งหมดของโพสต์นั้น และเปลี่ยนสถานะโพสต์สัมพันธ์
   - Security: authenticateToken + authorizeAdmin
---------------------------------------------------------------- */
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

          // 1) หา post_id จาก report เดียว
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

              // 2) อัปเดตตามสถานะใหม่
              if (newStatus === 'block') {
                  // รายงานทั้งหมดของโพสต์ -> 'block' และโพสต์ -> 'deactivate'
                  connection.query('UPDATE reports SET status = ? WHERE post_id = ?', ['block', postId], (errReports) => {
                      if (errReports) {
                          return connection.rollback(() => {
                              connection.release();
                              console.error('Error updating reports to block:', errReports);
                              res.status(500).json({ error: 'Failed to update reports: ' + errReports.message });
                          });
                      }
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
                  // รายงานทั้งหมดของโพสต์ -> 'normally' และโพสต์ -> 'active'
                  connection.query('UPDATE reports SET status = ? WHERE post_id = ?', ['normally', postId], (errReports) => {
                      if (errReports) {
                          return connection.rollback(() => {
                              connection.release();
                              console.error('Error updating reports to normally:', errReports);
                              res.status(500).json({ error: 'Failed to update reports: ' + errReports.message });
                          });
                      }
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
                  // รายงานทั้งหมดของโพสต์ -> 'pending' (ไม่ยุ่งโพสต์)
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


/* ----------------------------------------------------------------
   ADMIN: Get all categories
   - เรียงตาม CategoryID
   - Security: authenticateToken + authorizeAdmin
---------------------------------------------------------------- */
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


/* ----------------------------------------------------------------
   ADMIN: Create category
   - ต้องส่ง CategoryName
   - Security: authenticateToken + authorizeAdmin
---------------------------------------------------------------- */
app.post("/api/categories", authenticateToken, authorizeAdmin, (req, res) => {
  const { CategoryName } = req.body;

  if (!CategoryName) {
      return res.status(400).json({ error: "CategoryName is required" });
  }

  const createCategorySql = 'INSERT INTO category (CategoryName, created_at, updated_at) VALUES (?, NOW(), NOW())';
  pool.query(createCategorySql, [CategoryName], (err, results) => {
      if (err) {
          console.error("Database error during category creation:", err);
          return res.status(500).json({ error: "Error creating category" });
      }
      res.status(201).json({ message: "Category created successfully", categoryId: results.insertId });
  });
});


/* ----------------------------------------------------------------
   ADMIN: Update category by id
   - ต้องส่ง CategoryName
   - Security: authenticateToken + authorizeAdmin
---------------------------------------------------------------- */
app.put("/api/categories/:id", authenticateToken, authorizeAdmin, (req, res) => {
  const { id } = req.params;
  const { CategoryName } = req.body;

  if (!CategoryName) {
      return res.status(400).json({ error: "CategoryName is required" });
  }

  const updateCategorySql = 'UPDATE category SET CategoryName = ?, updated_at = NOW() WHERE CategoryID = ?';
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


/* ----------------------------------------------------------------
   ADMIN: Delete category by id
   - Security: authenticateToken + authorizeAdmin
---------------------------------------------------------------- */
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


/* ----------------------------------------------------------------
   ADMIN: Deactivate post and remove related reports
   - Body: { id } (post id)
   - ใช้ Transaction
   - Security: authenticateToken + authorizeAdmin
---------------------------------------------------------------- */
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
              // ตรวจสอบว่ามีรายงานของโพสต์นี้ไหม (ใช้ตรรกะเดิม)
              const checkPostInReportsSql = "SELECT * FROM reports WHERE post_id = ?";
              const [checkResults] = await connection.execute(checkPostInReportsSql, [id]);

              if (checkResults.length === 0) {
                  await connection.rollback();
                  connection.release();
                  return res.status(404).json({ error: "Post not found in pending reports (or already handled)" });
              }

              // อัปเดตสถานะโพสต์ -> 'deactivate'
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

              // ลบรายงานที่เกี่ยวข้องทั้งหมด
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


/* ----------------------------------------------------------------
   STATIC: serve slip images
---------------------------------------------------------------- */
app.use('/api/Slip', express.static(path.join(__dirname, 'Slip')));


/* ----------------------------------------------------------------
   ADMIN: Get all orders
   - join โฆษณาใหม่/ต่ออายุ แสดงข้อมูลรวม
   - Security: authenticateToken + authorizeAdmin
---------------------------------------------------------------- */
app.get("/api/admin/orders", authenticateToken, authorizeAdmin, (req, res) => {
  const sql = `
    SELECT
      o.*,
      COALESCE(a_new.title,  a_renew.title)   AS title,
      COALESCE(a_new.content,a_renew.content) AS content,
      COALESCE(a_new.link,   a_renew.link)    AS link,
      COALESCE(a_new.image,  a_renew.image)   AS image,
      COALESCE(a_new.status, a_renew.status)  AS ad_status
    FROM orders o
    LEFT JOIN ads a_new   ON a_new.order_id = o.id
    LEFT JOIN ads a_renew ON a_renew.id     = o.renew_ads_id
    ORDER BY o.created_at DESC
  `;
  pool.query(sql, (err, results) => {
    if (err) {
      console.error('[ERROR] Database error fetching all orders:', err);
      return res.status(500).json({ error: 'Database error' });
    }
    res.json(results);
  });
});


/* ----------------------------------------------------------------
   ADMIN: Update order and optional related ad
   - Body รองรับ: amount, status, prompay_number, title, content, link, image, expiration_date, admin_notes
   - ถ้า status = 'rejected' ต้องมี admin_notes
   - Transaction ครอบการอัปเดตทั้งหมด
   - Security: authenticateToken + authorizeAdmin
---------------------------------------------------------------- */
app.put("/api/admin/orders/:orderId", authenticateToken, authorizeAdmin, async (req, res) => {
  const { orderId } = req.params;
  const { amount, status, prompay_number, title, content, link, image, expiration_date, admin_notes } = req.body;

  // ตรวจเหตุผลเมื่อ reject
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

      // 1) อัปเดตตาราง orders (เฉพาะฟิลด์ที่ส่งมา)
      const updateOrderSql = `UPDATE orders SET amount = COALESCE(?, amount), status = COALESCE(?, status), prompay_number = COALESCE(?, prompay_number), updated_at = NOW() WHERE id = ?`;
      connection.query(updateOrderSql, [amount, status, prompay_number, orderId], (err, orderResult) => {
        if (err) {
          return connection.rollback(() => {
            connection.release();
            res.status(500).json({ error: 'Failed to update order.' });
          });
        }

        // 2) ถ้ามี field ของ ads ส่งมาด้วย ให้ update ads ที่สัมพันธ์กับ order นี้
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
              // แจ้งเตือนผู้ใช้เรื่องสถานะโฆษณา ถ้ามี status
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

        // 3) ถ้าไม่ได้แก้ ads ก็ commit แค่ออเดอร์
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


/* ----------------------------------------------------------------
   ADMIN: Delete order and related ads
   - ลบ ads ที่ผูกกับ order และตัว order เอง
   - ลบไฟล์ภาพ ads ในดิสก์หลัง commit
   - Security: authenticateToken + authorizeAdmin
---------------------------------------------------------------- */
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

      // 1) เตรียม path รูปจาก ads เพื่อไปลบไฟล์ทีหลัง
      const fetchAdSql = 'SELECT image FROM ads WHERE order_id = ?';
      connection.query(fetchAdSql, [orderId], (fetchErr, adResults) => {
        if (fetchErr) {
          return connection.rollback(() => {
            connection.release();
            res.status(500).json({ error: 'Failed to fetch ad for deletion.' });
          });
        }

        const imagePaths = adResults.map(row => row.image).filter(Boolean);

        // 2) ลบ ads ก่อน
        const deleteAdSql = 'DELETE FROM ads WHERE order_id = ?';
        connection.query(deleteAdSql, [orderId], (adDelErr, adDelResult) => {
          if (adDelErr) {
            return connection.rollback(() => {
              connection.release();
              res.status(500).json({ error: 'Failed to delete ad.' });
            });
          }

          // 3) แล้วค่อยลบ order
          const deleteOrderSql = 'DELETE FROM orders WHERE id = ?';
          connection.query(deleteOrderSql, [orderId], (orderDelErr, orderDelResult) => {
            if (orderDelErr) {
              return connection.rollback(() => {
                connection.release();
                res.status(500).json({ error: 'Failed to delete order.' });
              });
            }

            // 4) Commit แล้วค่อยลบไฟล์จากดิสก์
            connection.commit(commitErr => {
              if (commitErr) {
                return connection.rollback(() => {
                  connection.release();
                  res.status(500).json({ error: 'Transaction commit failed.' });
                });
              }
              connection.release();

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

/* ----------------------------------------------------------------
   ADMIN SEARCH: ads
   - auth: authenticateToken
   - q ใน query string, ค้น title/content/link/status/id (case-insensitive)
   - คืนรายการ ads เรียง created_at DESC
---------------------------------------------------------------- */
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

      // ไม่แก้ path image ใดๆ ส่งค่าตามที่เก็บไว้
      const adsWithImagePaths = results.map(ad => ({
        ...ad,
        image: ad.image ? ad.image : null
      }));

      res.status(200).json(adsWithImagePaths);
    }
  );
});


/* ----------------------------------------------------------------
   ADMIN SEARCH: users
   - auth: authenticateToken
   - q ค้น email/username/gender/bio/status/role/id (case-insensitive)
   - คืน picture เป็น string หรือ null ตามเดิม
---------------------------------------------------------------- */
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

      // map รูปให้เป็น string หรือ null (ไม่แตะ path เพิ่ม)
      const usersWithImagePaths = results.map(user => ({
        ...user,
        picture: user.picture && user.picture.trim() !== ''
          ? `${user.picture}`
          : null
      }));

      console.log("search results:", usersWithImagePaths);

      if (usersWithImagePaths.length === 0) {
        return res.status(200).json({ message: "No users found", results: [] });
      }

      res.status(200).json(usersWithImagePaths);
    }
  );
});


/* ----------------------------------------------------------------
   ADMIN SEARCH: posts
   - auth: authenticateToken
   - q ค้น content/Title/ProductName/status/id/user_id (case-insensitive)
   - ส่งข้อมูลโพสต์ตรงๆ
---------------------------------------------------------------- */
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

      res.status(200).json(results);
    }
  );
});


/* ----------------------------------------------------------------
   ADMIN SEARCH: reports
   - auth: authenticateToken
   - q ค้น reason/status/reportId/userId/postId/username/postTitle
   - join users & posts เพื่อได้ข้อมูลที่ UI ใช้
   - แปลง photo_url ถ้าเป็น JSON string
---------------------------------------------------------------- */
app.get("/api/admin/search/reports", authenticateToken, (req, res) => {
  const { q: query } = req.query;

  if (!query) {
    return res.status(400).json({ error: 'Search query is required' });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`;

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

      // แปลงรูปฟิลด์ photo_url เป็น array ถ้าเก็บเป็น JSON string
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

/* ----------------------------------------------------------------
   FOLLOW -> AUTO MATCH
   - สร้างความสัมพันธ์ follow และสร้าง match สำหรับแชทอัตโนมัติ
   - ป้องกันซ้ำด้วย ON DUPLICATE KEY
---------------------------------------------------------------- */
app.post("/api/users/:userId/follow/:followingId", (req, res) => {
    const { userId, followingId } = req.params;

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

        const createMatchQuery = `
            INSERT INTO matches (user1ID, user2ID, matchDate)
            VALUES (?, ?, NOW())
            ON DUPLICATE KEY UPDATE matchDate = matchDate
        `;

        pool.query(createMatchQuery, [userId, followingId], (err, matchResult) => {
            if (err) {
                console.error('Error creating match:', err);
                // ไม่ return error เพราะ follow สำเร็จแล้ว
            }
            
            res.status(200).json({ 
                message: 'Followed successfully',
                matchID: matchResult ? matchResult.insertId : null
            });
        });
    });
});


/* ----------------------------------------------------------------
   CREATE MATCH (MANUAL)
   - ใช้กรณีอยากยิงสร้าง match แยกภายหลังจาก follow แล้ว
   - ตรวจสอบมี follow จริง, กันสร้างซ้ำ
---------------------------------------------------------------- */
app.post("/api/create-match-on-follow", (req, res) => {
    const { followerID, followingID } = req.body;

    if (!followerID || !followingID) {
        return res.status(400).json({ error: 'Missing followerID or followingID' });
    }

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


/* ----------------------------------------------------------------
   LIST MATCHES FOR USER
   - ดึงรายการคู่แชท, รูป, ข้อความล่าสุด, เวลาปฏิสัมพันธ์ล่าสุด
   - เคารพการลบฝั่งเดียว (deleted_chats) และสถานะ follow/block
---------------------------------------------------------------- */
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


/* ----------------------------------------------------------------
   GET CHATS BY MATCH
   - โหลดข้อความทั้งหมดในแมตช์นั้น (เรียงเวลา ASC)
   - เติม URL รูปโปรไฟล์ให้เต็มถ้าเป็น path ภายใน
---------------------------------------------------------------- */
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

        return res.status(200).json({ messages: results });
    });
});


/* ----------------------------------------------------------------
   SEND MESSAGE
   - ตรวจสิทธิ์ผู้ส่งอยู่ใน match, ตรวจบล็อก, แล้วค่อย insert
---------------------------------------------------------------- */
app.post("/api/chats/:matchID", (req, res) => {
    const { matchID } = req.params;
    const { senderID, message } = req.body;

    if (!senderID || !message) {
        return res.status(400).json({ error: 'Missing senderID or message' });
    }

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


/* ----------------------------------------------------------------
   HIDE CHAT (ONE-SIDE DELETE)
   - mark ลบฝั่งเดียวใน deleted_chats พร้อม timestamp
---------------------------------------------------------------- */
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


/* ----------------------------------------------------------------
   RESTORE ALL CHATS (ONE USER)
   - ลบ record ใน deleted_chats ของ user นั้นทั้งหมด
---------------------------------------------------------------- */
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


/* ----------------------------------------------------------------
   BLOCK CHAT
   - ผู้ใช้ใน match เท่านั้นที่บล็อกได้
   - มีทั้ง insert ใหม่หรือ update flag เดิม
---------------------------------------------------------------- */
app.post("/api/block-chat", (req, res) => {
    const { userID, matchID, isBlocked } = req.body;

    if (!userID || !matchID || isBlocked === undefined) {
        return res.status(400).json({ error: 'Missing userID, matchID, or isBlocked' });
    }

    const matchQuery = `SELECT user1ID, user2ID FROM matches WHERE matchID = ?`;
    
    pool.query(matchQuery, [matchID], (err, results) => {
        if (err || results.length === 0) {
            console.error('Database error or match not found');
            return res.status(500).json({ error: 'Match not found or database error' });
        }

        const { user1ID, user2ID } = results[0];
        
        if (userID != user1ID && userID != user2ID) {
            return res.status(403).json({ error: 'User not authorized to block this chat' });
        }

        const blockerID = userID;
        const blockedID = (userID == user1ID) ? user2ID : user1ID;

        const checkQuery = `SELECT blockID FROM blocked_chats WHERE matchID = ? AND user1ID = ?`;
        
        pool.query(checkQuery, [matchID, blockerID], (err, checkResult) => {
            if (err) {
                console.error('Database error:', err);
                return res.status(500).json({ error: 'Database error' });
            }

            if (checkResult.length > 0) {
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


/* ----------------------------------------------------------------
   UNBLOCK CHAT
   - ปลดบล็อกโดยตั้ง isBlocked = 0 ของผู้บล็อกเดิม
---------------------------------------------------------------- */
app.post("/api/unblock-chat", (req, res) => {
    const { userID, matchID } = req.body;

    if (!userID || !matchID) {
        return res.status(400).json({ error: 'Missing userID or matchID' });
    }

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


/* ----------------------------------------------------------------
   CHECK BLOCK STATUS
   - ตรวจว่าผู้ใช้บล็อกเองหรือถูกอีกฝั่งบล็อกอยู่ใน match นั้น
---------------------------------------------------------------- */
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
// POST /api/orders (รองรับทั้งการสร้างใหม่และการต่ออายุ)
app.post("/api/orders", upload.single('image'), (req, res) => {
    console.log('[INFO] Received POST /api/orders request');
    
    // เพิ่ม renew_ads_id เข้ามา
    const { user_id, package_id, title, content, link, prompay_number, ad_start_date, renew_ads_id } = req.body;
    const imageFile = req.file;

    console.log('Received Body:', req.body);
    if(renew_ads_id) console.log('This is a RENEWAL request for Ad ID:', renew_ads_id);

    // --- Validation ---
    // ถ้าเป็นการต่ออายุ (มี renew_ads_id) จะไม่ต้องการ title, content, image, ad_start_date
    if (!user_id || !package_id || !prompay_number) {
        if (imageFile) fs.unlinkSync(imageFile.path);
        return res.status(400).json({ error: 'Missing required fields' });
    }
    
    // ตรวจสอบข้อมูลที่จำเป็นสำหรับการสร้างโฆษณาใหม่เท่านั้น
    if (!renew_ads_id) {
        if (!title || !content) {
            if (imageFile) fs.unlinkSync(imageFile.path);
            return res.status(400).json({ error: 'Title and content are required for a new ad.' });
        }
        if (!imageFile) {
            return res.status(400).json({ error: 'Missing required image file for a new ad.' });
        }
        if (!ad_start_date) {
            if (imageFile) fs.unlinkSync(imageFile.path);
            return res.status(400).json({ error: 'Please select an ad start date for a new ad.' });
        }
    }
    
    pool.query('SELECT * FROM ad_packages WHERE package_id = ?', [package_id], (err, pkg) => {
        if (err || pkg.length === 0) {
            if (imageFile) fs.unlinkSync(imageFile.path);
            return res.status(err ? 500 : 400).json({ error: err ? 'Database error' : 'Invalid package' });
        }
        
        const amount = pkg[0].price;
        const duration = pkg[0].duration_days;

        const orderSql = `
          INSERT INTO orders (user_id, amount, status, created_at, updated_at, prompay_number, package_id, renew_ads_id)
          VALUES (?, ?, 'pending', NOW(), NOW(), ?, ?, ?)
        `;
        
        pool.query(orderSql, [user_id, amount, prompay_number, package_id, renew_ads_id || null], (err, result) => {
            if (err) {
                if (imageFile) fs.unlinkSync(imageFile.path);
                return res.status(500).json({ error: 'Database error creating order' });
            }
            
            const order_id = result.insertId;
            console.log(`[INFO] Order ID ${order_id} created with status 'pending'.`);

            // ถ้าเป็นการสร้างโฆษณาใหม่ (ไม่มี renew_ads_id) ให้ INSERT ลงตาราง ads ด้วย
            if (!renew_ads_id) {
                const imagePath = `/uploads/${imageFile.filename}`;
                const adSql = `
                  INSERT INTO ads (user_id, order_id, title, content, link, image, status, show_at, created_at, expiration_date)
                  VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, NOW(), DATE_ADD(?, INTERVAL ? DAY))
                `;
                pool.query(adSql, [user_id, order_id, title, content, link || '', imagePath, ad_start_date, ad_start_date, duration], (err2) => {
                    if (err2) {
                        console.error('[ERROR] Database error creating ad for order ID ' + order_id + ':', err2);
                        return res.status(500).json({ error: 'Database error creating ad' });
                    }
                    console.log(`[INFO] Ad created for Order ID ${order_id} with status 'pending'.`);
                    res.status(201).json({ order_id, amount, duration });
                });
            } else {
                // ถ้าเป็นการต่ออายุ เราจะสร้างแค่ Order ไม่ต้องสร้าง Ad ใหม่
                console.log(`[INFO] Renewal Order ID ${order_id} created for Ad ID ${renew_ads_id}.`);
                res.status(201).json({ order_id, amount, duration });
            }
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
      return res.status(400).json({ error: 'Please upload the payment slip' });
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
                      res.status(500).json({ error: 'Error checking order status' });
                  });
              }
              if (orderResults.length === 0) {
                  return connection.rollback(() => {
                      connection.release();
                      res.status(404).json({ error: 'Order not found' });
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
                              res.status(500).json({ error: 'Error checking ad status' });
                          });
                      }
                      if (adResults.length === 0) {
                          return connection.rollback(() => {
                              connection.release();
                              res.status(404).json({ error: 'Ad to renew not found' });
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
                          let errorMessage = 'Cannot upload slip because the order status is invalid';
                          if (adExpirationDate < today) {
                              errorMessage = 'Cannot renew because the ad has already expired';
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
                          res.status(400).json({ error: 'Cannot upload slip, please wait for admin to approve the content first' });
                      });
                  }
                  proceedUpdateOrderAndAd(connection, orderId, req.file.path, renewAdsId, packageId, null, null, canUpload, res); // ส่ง null สำหรับค่าที่ไม่เกี่ยวข้อง
              }
          });
      });
  });
});


// PATCH /api/ads/:adId/user-delete
app.patch('/api/ads/:adId/user-delete', async (req, res) => {
  const adId = Number(req.params.adId);
  if (!Number.isInteger(adId)) {
    return res.status(400).json({ error: 'Invalid ad id' });
  }

  const getConn = () => new Promise((resolve, reject) => {
    pool.getConnection((err, c) => (err ? reject(err) : resolve(c)));
  });
  const q = (c, sql, params) => new Promise((resolve, reject) => {
    c.query(sql, params, (err, r) => (err ? reject(err) : resolve(r)));
  });

  let conn;
  try {
    conn = await getConn();
    await q(conn, 'START TRANSACTION');

    // 1) อ่านข้อมูล ads + orders
    const rows = await q(
      conn,
      `
      SELECT a.id AS ad_id, a.status AS ad_status,
             o.id AS order_id, o.status AS order_status
      FROM ads a
      JOIN orders o ON o.id = a.order_id
      WHERE a.id = ?
      FOR UPDATE
      `,
      [adId]
    );

    if (rows.length === 0) {
      await q(conn, 'ROLLBACK');
      return res.status(404).json({ error: 'Ad not found' });
    }

    const { ad_status, order_status, order_id } = rows[0];
    const blocked = new Set(['paid', 'active']);
    if (blocked.has(ad_status) || blocked.has(order_status)) {
      await q(conn, 'ROLLBACK');
      return res.status(409).json({
        error: 'Cannot mark as deleted: status is paid or active',
        ad_status,
        order_status
      });
    }

    // 2) อัปเดต ads
    const updateAds = await q(
      conn,
      `UPDATE ads SET status='userdelete', updated_at=NOW() WHERE id=? LIMIT 1`,
      [adId]
    );
    if (updateAds.affectedRows !== 1) {
      await q(conn, 'ROLLBACK');
      return res.status(500).json({ error: 'Failed to update ads status' });
    }

    // 3) อัปเดต orders
    const updateOrders = await q(
      conn,
      `UPDATE orders SET status='userdelete', updated_at=NOW() WHERE id=? LIMIT 1`,
      [order_id]
    );
    if (updateOrders.affectedRows !== 1) {
      await q(conn, 'ROLLBACK');
      return res.status(500).json({ error: 'Failed to update orders status' });
    }

    await q(conn, 'COMMIT');
    return res.status(200).json({
      message: 'Ad and order marked as userdelete',
      ad_id: adId,
      order_id
    });
  } catch (err) {
    if (conn) try { await q(conn, 'ROLLBACK'); } catch {}
    console.error(err);
    return res.status(500).json({ error: 'Server error' });
  } finally {
    if (conn) conn.release();
  }
});


//########################################################   Ads API  ########################################################


// GET /api/my/ads  (User-only)
app.get('/api/my/ads', authenticateToken, async (req, res) => {
  const userId = req.user.id; // มาจาก token

  const {
    status,              // 'pending' | 'approved' | 'paid' | 'active' | 'rejected' | 'expired' | 'userdelete' | 'all'
    includeDeleted,      // 'true' เพื่อให้รวม userdelete
    from,                // 'YYYY-MM-DD' หรือ datetime
    to,                  // 'YYYY-MM-DD' หรือ datetime
    sort = 'created_at', // 'created_at' | 'show_at' | 'expiration_date'
    order = 'desc',      // 'asc' | 'desc'
    page = 1,
    limit = 20
  } = req.query;

  // ป้องกัน SQL injection ด้วย whitelist
  const sortMap = { created_at: 'a.created_at', show_at: 'a.show_at', expiration_date: 'a.expiration_date' };
  const sortCol = sortMap[sort] || 'a.created_at';
  const dir = (order || '').toLowerCase() === 'asc' ? 'ASC' : 'DESC';
  const pageNum = Math.max(parseInt(page, 10) || 1, 1);
  const pageSize = Math.min(Math.max(parseInt(limit, 10) || 20, 1), 100);
  const offset = (pageNum - 1) * pageSize;

  const where = ['a.user_id = ?'];
  const params = [userId];

  // ดีฟอลต์ไม่เอา userdelete
  const wantIncludeDeleted = String(includeDeleted).toLowerCase() === 'true';
  if (!wantIncludeDeleted) where.push(`a.status <> 'userdelete'`);

  // สถานะ: ถ้าระบุและไม่ใช่ 'all' ให้กรอง
  if (status && status !== 'all') {
    where.push('a.status = ?');
    params.push(status);
  }

  if (from) { where.push('a.created_at >= ?'); params.push(from); }
  if (to)   { where.push('a.created_at <= ?'); params.push(to); }

  const baseFromJoin = `
    FROM ads a
    LEFT JOIN orders o ON o.id = a.order_id
    LEFT JOIN ad_packages p ON p.package_id = o.package_id
    WHERE ${where.join(' AND ')}
  `;

  const listSql = `
    SELECT
      a.id, a.user_id, a.order_id, a.title, a.content, a.link, a.image,
      a.status,
      DATE_FORMAT(a.show_at, "%Y-%m-%d") AS show_at,  
      a.created_at, a.updated_at,
      DATE_FORMAT(a.expiration_date, "%Y-%m-%d") AS expiration_date, 
      a.display_count,
      o.amount, o.status AS order_status,
      p.name AS package_name, p.price AS package_price, p.duration_days AS package_duration
    ${baseFromJoin}
    ORDER BY
      FIELD(a.status,'pending','approved','paid','active','rejected','expired','userdelete'),
      ${sortCol} ${dir}
    LIMIT ? OFFSET ?;
  `;

  const countSql = `SELECT COUNT(*) AS total ${baseFromJoin};`;

  const getConn = () => new Promise((resolve, reject) => {
    pool.getConnection((e, c) => e ? reject(e) : resolve(c));
  });
  const q = (c, sql, p=[]) => new Promise((resolve, reject) => {
    c.query(sql, p, (e, r) => e ? reject(e) : resolve(r));
  });

  let conn;
  try {
    conn = await getConn();
    const [rows, countRows] = await Promise.all([
      q(conn, listSql, [...params, pageSize, offset]),
      q(conn, countSql, params)
    ]);
    const total = countRows[0]?.total || 0;

    res.json({
      data: rows,
      pagination: {
        total,
        page: pageNum,
        limit: pageSize,
        pages: Math.ceil(total / pageSize)
      }
    });
  } catch (err) {
    console.error('[GET /api/my/ads] Error:', err);
    res.status(500).json({ error: 'Error fetching user ads' });
  } finally {
    if (conn) conn.release();
  }
});



// PUT /api/my/ads/:adId/delete (User-only soft delete)   สำหรับลบโฆษณาในแอป
app.put('/api/my/ads/:adId/delete', authenticateToken, async (req, res) => {
    const { adId } = req.params;
    const userId = req.user.id;

    console.log(`[INFO] User ${userId} is requesting to delete Ad ID: ${adId}`);

    const sql = `
        UPDATE ads 
        SET status = 'userdelete' 
        WHERE id = ? AND user_id = ? AND status <> 'active'
    `;

    pool.query(sql, [adId, userId], (err, result) => {
        if (err) {
            console.error('[ERROR] Database error during soft delete:', err);
            return res.status(500).json({ error: 'Database error' });
        }
        
        if (result.affectedRows === 0) {
            // อาจจะหาโฆษณาไม่เจอ, ไม่ใช่เจ้าของ, หรือโฆษณา active อยู่
            return res.status(404).json({ error: 'Ad not found, you are not the owner, or the ad is already active and cannot be deleted.' });
        }

        console.log(`[SUCCESS] Ad ID ${adId} was soft-deleted by User ID ${userId}.`);
        res.status(200).json({ message: 'Ad has been deleted successfully.' });
    });
});


// GET /api/my/ads/:id  (ดึงโฆษณาตาม id)
app.get('/api/my/ads/:id', authenticateToken, async (req, res) => {
  const userId = req.user.id;
  const adId = req.params.id;

  const sql = `
    SELECT a.id, a.user_id, a.order_id, a.title, a.content, a.link, a.image,
           a.status,
           DATE_FORMAT(a.show_at, "%Y-%m-%d") AS show_at,  
           a.created_at, a.updated_at,
           DATE_FORMAT(a.expiration_date, "%Y-%m-%d") AS expiration_date, 
           a.display_count,
           o.amount, o.status AS order_status,
           p.name AS package_name, p.price AS package_price, p.duration_days AS package_duration
    FROM ads a
    LEFT JOIN orders o ON o.id = a.order_id
    LEFT JOIN ad_packages p ON p.package_id = o.package_id
    WHERE a.id = ? AND a.user_id = ?;
  `;

  pool.query(sql, [adId, userId], (err, results) => {
    if (err) {
      console.error('Error fetching ad detail:', err);
      return res.status(500).json({ error: 'Error fetching ad detail' });
    }
    if (results.length === 0) {
      return res.status(404).json({ error: 'Ad not found or you do not own it' });
    }
    res.json(results[0]);
  });
});


// GET /api/ad-packages
app.get("/api/ad-packages", (req, res) => {
    console.log('[INFO] Received GET /api/ad-packages request');
    
    // ✅ แก้ไข SQL จาก SELECT * ให้เป็นแบบนี้
    const sql = 'SELECT package_id AS id, name, price, duration_days FROM ad_packages ORDER BY duration_days ASC';
    
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
      return res.status(400).json({ error: 'Please fill in all required fields (package_id, prompay_number)' });
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
                      res.status(500).json({ error: 'Error deleting old order' });
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
                          res.status(404).json({ error: 'Ad not found or you do not have permission to renew this ad' });
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
                          res.status(400).json({ error: 'Cannot renew because the ad has already expired' });
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
                              res.status(400).json({ error: 'Selected package not found' });
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
                                  res.status(500).json({ error: 'Error creating renewal order' });
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
                                  message: 'Renewal order created successfully! Please make payment and upload the slip',
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