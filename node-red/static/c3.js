// 配置（必要时修改）
  const POLL_BASE = `${window.location.protocol}//${window.location.hostname}:1881`;
  const POLL_INTERVAL_MS = 200;
  const STEP_ADJUSTMENT = 10;

  $(function () {
    let currentZoom = null;
    let currentFocus = null;
    let currentMode = 'manual';
    let zoomStep = 40;
    let focusStep = 40;

    let autoAdjustRunning = false;  // 防止重复触发
    let adjustAttempts = 0;
    const MAX_ATTEMPTS = 10;
    const TARGET_MIN = 300;
    const TARGET_MAX = 380;

    function setStatus(text) {
      $('#status').text('状态：' + text);
    }

    function updateModeDisplay(mode) {
      currentMode = mode;
      if (mode === 'manual') {
        $('#manualBtn').addClass('active');
        $('#autoBtn').removeClass('active');
        $('#modeStatus').text('手动模式');
        $('#zoomIn, #zoomOut, #focusPlus, #focusMinus').prop('disabled', false);
      } else {
        $('#autoBtn').addClass('active');
        $('#manualBtn').removeClass('active');
        $('#modeStatus').text('自动模式');
        $('#zoomIn, #zoomOut, #focusPlus, #focusMinus').prop('disabled', true);
      }
    }

    function setMode(mode) {
      $.get(POLL_BASE + '/mode', { mode: mode })
        .done(() => {
          updateModeDisplay(mode);
          setStatus('模式已切换至：' + (mode === 'manual' ? '手动' : '自动'));
        })
        .fail(() => setStatus('模式切换失败'));
    }

    function pollValues() {
      $.get(POLL_BASE + '/data').done(function (data) {
        if (Array.isArray(data) && data.length >= 3) {
          let laserVal = parseNumericResponse(data[0]);
          if (laserVal !== null) $('#laserVal').text(laserVal);

          let zoomVal = parseNumericResponse(data[1]);
          if (zoomVal !== null) {
            currentZoom = zoomVal;
            $('#zoomVal').text(zoomVal);
          }

          let focusVal = parseNumericResponse(data[2]);
          if (focusVal !== null) {
            currentFocus = focusVal;
            $('#focusVal').text(focusVal);
          }
        }
      });
    }

    function parseNumericResponse(data) {
      try {
        if (typeof data === 'number') return data;
        if (typeof data === 'string') {
          let trimmed = data.trim();
          if (!trimmed) return null;
          if (trimmed[0] === '{' || trimmed[0] === '[') {
            let j = JSON.parse(trimmed);
            if (j && j.value !== undefined) return Number(j.value);
            for (let k in j) if (typeof j[k] === 'number') return j[k];
          }
          let n = parseFloat(trimmed);
          return isNaN(n) ? null : n;
        }
      } catch (e) { return null; }
      return null;
    }

    function sendSet(endpoint, value) {
      $.get(POLL_BASE + '/' + endpoint, { value: value }).done(() => {
        setStatus(endpoint + ' 已发送: ' + value);
        setTimeout(pollValues, 300); // 稍微延迟刷新
      });
    }

    function sendReadCamera() {
      return $.get(POLL_BASE + '/readCamera').done(() => {
        setStatus('相机自读参数已执行');
      });
    }

    function sendAutoFocus() {
      return $.ajax({
        url: POLL_BASE + '/auto_focus',
        method: 'GET',
        data: { cmd: 'auto_focus' },
        dataType: 'json'
      });
    }

    function zoomInOnce() {
      let target = currentZoom === null ? zoomStep : Number(currentZoom) + zoomStep;
      return sendSet('setzoom', target);
    }

    // 延迟函数
    const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

    // 核心：自动迭代调整函数
    async function startAutoAdjust() {
      if (autoAdjustRunning) {
        setStatus('自动调整已在进行中...');
        return;
      }

      autoAdjustRunning = true;
      adjustAttempts = 0;
      setStatus('开始自动优化圆形大小（目标 300-380）');

      while (adjustAttempts < MAX_ATTEMPTS) {
        adjustAttempts++;
        setStatus(`第 ${adjustAttempts} 次分析中...`);

        try {
          const response = await $.ajax({
            url: POLL_BASE + '/detect',
            method: 'GET',
            data: { cmd: 'detect' },
            dataType: 'json',
            timeout: 10000
          });

          console.log('检测结果:', response);

          if (!response || !response.circle || typeof response.circle.diameter !== 'number') {
            setStatus(`第 ${adjustAttempts} 次：未检测到圆形`);
            break;
          }

          const diameter = Math.round(response.circle.diameter);
          setStatus(`第 ${adjustAttempts} 次：直径 ${diameter}px`);

          // 成功条件：直径在目标范围内
          if (diameter >= TARGET_MIN && diameter <= TARGET_MAX) {
            setStatus(`成功！直径 ${diameter}px 在目标范围内 (300-380)`);
            $('#message').text(`优化成功！直径：${diameter}px`).css({color: '#4CAF50', fontWeight: 'bold'});
            autoAdjustRunning = false;
            return;
          }

          // 如果太小，才执行调整流程
          if (diameter >= 300) {
            setStatus(`直径 ${diameter}px 已较大，停止调整`);
            break;
          }

          // —— 开始自动调整流程 ——
          setStatus(`直径 ${diameter}px 过小，开始第 ${adjustAttempts} 次调整...`);

          await sendReadCamera();           // 1. 相机自读
          await delay(1500);

          await zoomInOnce();               // 2. 变焦放大
          await delay(1000);

          await sendAutoFocus();            // 3. 第一次自动聚焦
          await delay(800);
          await sendAutoFocus();            // 4. 第二次自动聚焦
          await delay(2000);

        } catch (err) {
          console.error('分析或控制出错:', err);
          setStatus('通信异常，中止自动优化');
          break;
        }
      }

      // 循环结束
      if (adjustAttempts >= MAX_ATTEMPTS) {
        setStatus('已达最大尝试次数 '+MAX_ATTEMPTS+' 次，优化失败');
        $('#message').text('优化失败：无法使圆达到目标大小').css({color: '#f44336', fontWeight: 'bold'});
      }
      else{
         $.get(POLL_BASE + '/auto_done').done(function (data) {
            setStatus('已记录点位');
         });
      }

      autoAdjustRunning = false;
    }

    // ==================== 事件绑定 ====================

    $('#manualBtn, #autoBtn').on('click', function () {
      setMode($(this).attr('id') === 'manualBtn' ? 'manual' : 'auto');
    });

    $('#do_zoom').on('click', function () {
      $.ajax({
        url: POLL_BASE + '/do_zoom',
        method: 'GET',
        data: { cmd: 'do_zoom' },
        dataType: 'json'
      }).done(startAutoAdjust); // 可选：标定后直接开始优化
    });

    $('#auto_focus').on('click', () => sendAutoFocus());

    $('#zoomStepPlus, #zoomStepMinus, #focusStepPlus, #focusStepMinus').on('click', function () {
      const isZoom = this.id.includes('zoom');
      const isPlus = this.id.includes('Plus');
      let target = isZoom ? zoomStep : focusStep;
      if (isPlus) target += STEP_ADJUSTMENT;
      else if (target > STEP_ADJUSTMENT) target -= STEP_ADJUSTMENT;
      if (isZoom) zoomStep = target; else focusStep = target;
      $(isZoom ? '#zoomStepVal' : '#focusStepVal').text(target);
      setStatus((isZoom ? '变焦' : '聚焦') + '步长已调整为: ' + target);
    });

    $('#zoomIn').on('click', () => {
      let newVal = currentZoom === null ? zoomStep : Number(currentZoom) + zoomStep;
      sendSet('setzoom', newVal);
    });

    $('#zoomOut').on('click', () => {
      let newVal = currentZoom === null ? 0 : Math.max(0, Number(currentZoom) - zoomStep);
      sendSet('setzoom', newVal);
    });

    $('#focusPlus').on('click', () => {
      let newVal = currentFocus === null ? focusStep : Number(currentFocus) + focusStep;
      sendSet('setfocus', newVal);
    });

    $('#focusMinus').on('click', () => {
      let newVal = currentFocus === null ? 0 : Math.max(0, Number(currentFocus) - focusStep);
      sendSet('setfocus', newVal);
    });

    $('#readCamera').on('click', sendReadCamera);

    // 关键：点击“图片分析”按钮 → 启动自动优化流程
    $('#detectBtn').off('click').on('click', function () {
      if (autoAdjustRunning) {
        setStatus('自动优化进行中，请勿重复点击');
        return;
      }
      $('#message').text('');
      startAutoAdjust();
    });

    // 启动轮询
    pollValues();
    setInterval(pollValues, POLL_INTERVAL_MS);
    setStatus('运行中，自动优化功能已就绪');
  });