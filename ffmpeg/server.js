const express = require("express");
const { spawn } = require("child_process");
const app = express();

const PORT = 8090;
const RTSP_URL = "rtsp://admin:gene2025@192.168.1.64/ISAPI/Streaming/channels/102";

app.get("/mjpeg", (req, res) => {
  console.log("New MJPEG client connected");
  
  res.writeHead(200, {
    "Content-Type": "multipart/x-mixed-replace; boundary=ffserver",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
    "Connection": "close",
    "Access-Control-Allow-Origin": "*"
  });

  const ffmpeg = spawn("ffmpeg", [
    "-rtsp_transport", "tcp",
    "-i", RTSP_URL,
    "-f", "mjpeg",
    "-q:v", "5",
    "-r", "10",
    "-vf", "scale=-2:480", // 可选：缩放视频以减少带宽
    "-"
  ]);

  let isClientConnected = true;

  ffmpeg.stdout.on("data", (chunk) => {
    if (!isClientConnected) {
      return;
    }
    
    try {
      // 确保响应还没有结束
      if (!res.headersSent || res.destroyed) {
        return;
      }
      
      // 写入MJPEG边界和头部
      const boundary = `--ffserver\r\nContent-Type: image/jpeg\r\nContent-Length: ${chunk.length}\r\n\r\n`;
      res.write(boundary);
      res.write(chunk);
      res.write("\r\n"); // 添加边界结尾
      
    } catch (err) {
      console.log("Error writing to client:", err.message);
      cleanup();
    }
  });

  ffmpeg.stderr.on("data", (data) => {
    // 可以取消注释来调试FFmpeg输出
    // console.log(`FFmpeg stderr: ${data}`);
  });

  ffmpeg.on("error", (err) => {
    console.log("FFmpeg error:", err.message);
    cleanup();
  });

  ffmpeg.on("close", (code) => {
    console.log(`FFmpeg process exited with code ${code}`);
    cleanup();
  });

  const cleanup = () => {
    if (isClientConnected) {
      isClientConnected = false;
      console.log("Cleaning up FFmpeg process");
      
      try {
        if (!ffmpeg.killed) {
          ffmpeg.kill("SIGTERM");
          // 如果SIGTERM没有工作，5秒后强制杀死
          setTimeout(() => {
            if (!ffmpeg.killed) {
              ffmpeg.kill("SIGKILL");
            }
          }, 5000);
        }
      } catch (err) {
        console.log("Error killing FFmpeg:", err.message);
      }
      
      try {
        if (!res.headersSent) {
          res.status(500).end();
        } else if (!res.destroyed) {
          res.end();
        }
      } catch (err) {
        console.log("Error ending response:", err.message);
      }
    }
  };

  // 处理客户端断开连接
  req.on("close", () => {
    console.log("Client disconnected");
    cleanup();
  });

  req.on("error", (err) => {
    console.log("Request error:", err.message);
    cleanup();
  });

  res.on("error", (err) => {
    console.log("Response error:", err.message);
    cleanup();
  });

  // 超时处理（可选）
  const timeout = setTimeout(() => {
    console.log("Connection timeout");
    cleanup();
  }, 300000); // 5分钟超时

  req.on("close", () => {
    clearTimeout(timeout);
  });
});

// 添加一个简单的HTML页面来测试流
app.get("/", (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>MJPEG Stream Test</title>
    </head>
    <body>
        <h1>MJPEG Stream Test</h1>
        <img src="/mjpeg" style="max-width: 100%; height: auto;" />
        <br>
        <button onclick="location.reload()">刷新流</button>
    </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log(`MJPEG server running at http://localhost:${PORT}`);
  console.log(`Direct stream: http://localhost:${PORT}/mjpeg`);
});