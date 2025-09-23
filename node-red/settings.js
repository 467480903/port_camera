module.exports = {
    // 流文件存放目录
    flowFile: 'flows.json',
    
    // 用户目录
    userDir: 'D:\\projects2025\\camera\\port_camera\\node-red',
    
    // 管理员认证
    // adminAuth: {
    //     type: "credentials",
    //     users: [{
    //         username: "admin",
    //         password: "$2a$08$zZWtXTja0fB1pzD4sHCMyOCMYz2Z6dNbM6tl8sJogENOMcxWV9DN.",
    //         permissions: "*"
    //     }]
    // },
    
    // HTTP 节点配置
    httpNodeRoot: '/',

    httpStatic: 'static',
    
    // 端口设置
    uiPort: 1881,
    
    // 调试输出
    logging: {
        console: {
            level: "info",
            metrics: false,
            audit: false
        }
    }
};