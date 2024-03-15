var objPath = './res.obj';
var scale = 2.5;
var INF = 100000000;

var camera, scene, renderer, controls, animationId;

var mouseX = 0, mouseY = 0;

var xCenter, yCenter, zCenter, zOffset;

$(document).ready(function () {
    $('#btn-upload')[0].disabled = false;
    $('#btn-default')[0].disabled = false;
});


function initScene() {

    var container = $("#container")[0];

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.x = 0;
    camera.position.y = 0;
    camera.position.z = -zOffset;

    // cameraQuaternion = new THREE.Quaternion();
    // cameraQuaternion.set(1, 0, 0, 1);
    // camera.quaternion.multiply(cameraQuaternion);
    // camera.rotation.setFromQuaternion(camera.quaternion, camera.rotation.order);
    // camera.rotation.x = 1;
    // camera.updateProjectionMatrix();

    
    // scene

    scene = new THREE.Scene();

    var ambientLight = new THREE.AmbientLight(0xcccccc, 0.3);
    scene.add(ambientLight);

    var pointLight = new THREE.PointLight(0xeeeeee, 0.6);
    camera.add(pointLight);
    scene.add(camera);

    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth / 1.3, window.innerHeight / 1.3);
    // renderer.setSize(800, 510);
    renderer.domElement.id = "scene";
    container.appendChild(renderer.domElement);

    // document.addEventListener('mousemove', onDocumentMouseMove, false);
    // window.addEventListener('resize', onWindowResize, false);

    controls = new THREE.TrackballControls(camera, renderer.domElement);
    controls.dynamicDampingFactor = 0.5;
    controls.rotateSpeed = 2;
    controls.zoomSpeed = 2;
    controls.panSpeed = 3;

}

function clearScene() {
    $("#stats").html("");
    $("#scene").remove();
    cancelAnimationFrame(animationId);
}

function showOBJ(data) {

    var vertexCnt = 0, faceCnt = 0;
    var xMin = INF, yMin = INF, zMin = INF;
    var xMax = -INF, yMax = -INF, zMax = -INF;

    var dataLines = data.split('\n');
    for (var i = 0; i < dataLines.length; i++) {
        if (dataLines[i][0] == 'v' && dataLines[i][1] == ' ') {
            vertexCnt++;
            var vertexData = dataLines[i].split(' ');
            xMin = Math.min(xMin, vertexData[1]);
            xMax = Math.max(xMax, vertexData[1]);
            yMin = Math.min(yMin, vertexData[2]);
            yMax = Math.max(yMax, vertexData[2]);
            zMin = Math.min(zMin, vertexData[3]);
            zMax = Math.max(zMax, vertexData[3]);

        } else if (dataLines[i][0] == 'f' && dataLines[i][1] == ' ') {
            faceCnt++;
        } else {
            continue;
        }
    }

    xCenter = (xMin + xMax) / 2;
    yCenter = (yMin + yMax) / 2;
    zCenter = (zMin + zMax) / 2;
    zOffset = (zMax - zMin) * scale;

    console.log(xMin, xMax, yMin, yMax, zMin, zMax);
    $("#vertex-cnt").html("Vertex count: " + vertexCnt);
    $("#face-cnt").html("Triangle face count: " + faceCnt);

    initScene();

    var loader = new THREE.OBJLoader();
    mesh = loader.parse(data).children[0];
    mesh.geometry.translate(-xCenter, -yCenter, -zCenter);
    mesh.geometry.rotateZ(Math.PI);
    mesh.material.side = THREE.DoubleSide;
    scene.add(mesh);

    animate();
}
function uploadData() {
    console.log('Upload data');

    clearScene();

    var objFile = $("#obj-file")[0].files[0];

    var fileReader = new FileReader();

    fileReader.onload = function () {
        $('#stats').html("obj file loaded.");
        showOBJ(fileReader.result);
    };

    fileReader.onprogress = function (data) {
        if (data.lengthComputable) {
            var progress = parseInt(data.loaded / data.total * 100);
            $("#stats").html('obj ' + Math.round(progress, 2) + '% loaded');
        }
    };

    fileReader.onerror = function (err) {
        console.error("[ERR] An error happened when loading file.", err);
    }

    fileReader.readAsText(objFile);

}

function downloadData() {
    console.log('download data');
    $.ajax({
        type: "get",       
        url:"http://10.33.30.37:8899/download",
        data: {filename : objPath},
        async: true,
        responseType: 'blob',
        success: function (res) {
            let blob = new Blob([res])
            let reader = new FileReader()
            reader.readAsDataURL(blob)
            reader.onload = (e) => {
            let a = document.createElement('a')
            /* 默认文件名 */
            a.download = 'result.obj'
            a.href = e.target.result
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            this.onClose()
          }
        },
	    error: function(data) {
            alert('下载失败请重试'); //alert错误信息 
        }
    });
}

function freshData() {
    console.log('fresh data');
    $('#down').css('display', 'none')
    $('#fresh').css('display', 'none')
    $('#scene').css('display', 'none')
    $('#log').css('display', 'block')
}

function useDefaultData() {
    console.log('Use default data.')

    clearScene();

    // Load obj file
    var fileLoader = new THREE.FileLoader();

    fileLoader.load(

        'obj/' + objPath + '.obj', // resource URL

        function (data) { // onLoad callback
            // $('#stats').html("obj file loaded.");
            showOBJ(data);
        },

        function (xhr) { // onProgress callback
            var progress = Math.floor(xhr.loaded / xhr.total * 100);
            // if (!isNaN(progress)) {
            //     $("#stats").html('obj ' + Math.round(progress, 2) + '% loaded');
            // }
        },

        function (err) { // onError callback
            console.error("[ERR] An error happened when loading " + objPath + ".", err);
        }

    );
}


function onWindowResize() {

    windowHalfX = window.innerWidth / 2;
    windowHalfY = window.innerHeight / 2;

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);

    controls.handleResize();

}


function animate() {

    animationId = requestAnimationFrame(animate);
    
    controls.update();
    renderer.render(scene, camera);

}

// 定义画布
var canvas = document.getElementById('canvas')
var is_eraser = false
// 画板大小
canvas.width = 700
canvas.height = 600

var context = canvas.getContext('2d')
// 背景颜色
context.fillStyle = 'white'
context.fillRect(0, 0, canvas.width, canvas.height)

// 线宽提示
var range = document.getElementById('customRange1')
range.oninput = function () {
    this.title = 'lineWidth: ' + this.value
}
var rect = canvas.getBoundingClientRect()
var x = (0 - rect.left) / rect.width * canvas.width
var y = (0 - rect.top) / rect.height * canvas.height
var Mouse = { x: x, y: y }
var lastMouse = { x: x, y: y }
var painting = false

var example = document.getElementById('example')
example.onclick = function () {
    var img = new Image();
    var i = parseInt((Math.random()*(9-1)+1))
    img.src = 'img/example/t (' + i + ').jpg';
    // 图片跨域访问
    img.setAttribute('crossOrigin', 'anonymous');
    img.onload = function() {
        context.drawImage(img, 0, 0, 700, 600);
    };
}

var input = document.getElementById('input');
input.addEventListener('change', handleFiles);

function handleFiles(e) {
    var img = new Image
    img.src = URL.createObjectURL(e.target.files[0])
    img.onload = function() {
        context.drawImage(img, 0, 0, 700, 600);
    }
}

var upload = document.getElementById('upload')
upload.onclick = function (){
    input.click()
}

var eraser = document.getElementById('eraser')
eraser.onclick = function () {
    is_eraser = ! is_eraser
}
//var upload = document.getElementById('upload')
//upload.onclick = function (){
//    var input = document.getElementById('in')
//    input.click()
//    file = input.files[0]
//    var img = new Image
//    img.src = URL.createObjectURL(file)
//    img.onload = function() {
//        context.drawImage(img, 0, 0, 400, 300);
//    }
//}


canvas.onmousedown = function () {
    painting = true
}

canvas.onmousemove = function (a) {
    var color = 'black'
    var value = range.value
    if(is_eraser){
        color = 'white'
        value = range.value
    }
    var rect = canvas.getBoundingClientRect()
    var x = (a.clientX - rect.left) / rect.width * canvas.width
    var y = (a.clientY - rect.top) / rect.height * canvas.height
    lastMouse.x = Mouse.x
    lastMouse.y = Mouse.y
    Mouse.x = x
    Mouse.y = y
    if (painting) {
        /*
        画笔参数：
            linewidth: 线宽
            lineJoin: 线条转角的样式, 'round': 转角是圆头
            lineCap: 线条端点的样式, 'round': 线的端点多出一个圆弧
            strokeStyle: 描边的样式, 'white': 设置描边为白色
        */
        context.lineWidth = value
        context.lineJoin = 'round'
        context.lineCap = 'round'
        context.strokeStyle = color

        // 开始绘画
        context.beginPath()
        if(!is_eraser){
            context.moveTo(lastMouse.x, lastMouse.y);
            context.lineTo(Mouse.x, Mouse.y);
        }
        else{
            context.fillStyle = color
            context.arc(x, y, value, 0, 2 * Math.PI);
            context.fill()
        }
        context.closePath()
        context.stroke()
    }

}
canvas.onmouseup = function () {
    painting = false
}

canvas.ontouchstart = function(a){
    painting = true
    Mouse.x = (a.touches[0].clientX - rect.left) / rect.width * canvas.width
    Mouse.y = (a.touches[0].clientY - rect.top) / rect.height * canvas.height
    lastMouse.x = Mouse.x
    lastMouse.y = Mouse.y
}

canvas.ontouchmove = function(a){
    var color = 'black'
    var value = range.value
    if(is_eraser){
        color = 'white'
        value = range.value * 10
    }
    var rect = canvas.getBoundingClientRect()
    if(!painting){
        painting = true
        lastMouse.x = (a.touches[0].clientX - rect.left) / rect.width * canvas.width
        lastMouse.y = (a.touches[0].clientY - rect.top) / rect.height * canvas.height
    }
    lastMouse.x = Mouse.x
    lastMouse.y = Mouse.y
    Mouse.x = (a.touches[0].clientX - rect.left) / rect.width * canvas.width
    Mouse.y = (a.touches[0].clientY - rect.top) / rect.height * canvas.height
//    }
    if (painting) {
        /*
        画笔参数：
            linewidth: 线宽
            lineJoin: 线条转角的样式, 'round': 转角是圆头
            lineCap: 线条端点的样式, 'round': 线的端点多出一个圆弧
            strokeStyle: 描边的样式, 'white': 设置描边为白色
        */
        context.lineWidth = value
        context.lineJoin = 'round'
        context.lineCap = 'round'
        context.strokeStyle = color

        // 开始绘画
        context.beginPath()
        context.moveTo(lastMouse.x, lastMouse.y);
        context.lineTo(Mouse.x, Mouse.y);
//        else{
//            context.fillStyle = color
//            context.arc(x, y, value, 0, 2 * Math.PI);
//            context.fill()
//        }
        context.closePath()
        context.stroke()
    }
}
canvas.addEventListener("touchstart", function(){
    painting = false
}, false);


// 下载图片
var downl = document.getElementById('downl')
downl.onclick = function () {
    objPath = Date.now()
    // 获取Canvas的编码。
    var imgData = document.getElementById("canvas").toDataURL("image/jpeg");
    // 上传到后台。
    $.ajax({
        type: "post",       
        url:"http://10.33.30.37:8899/construct",
        data: {image : imgData, filename : objPath},
        async: true,
        success: function (htmlVal) {
        useDefaultData()
        $('#down').css('display', 'inline')
        $('#fresh').css('display', 'inline')
        $('#log').css('display', 'none')
        // alert("可以下载三维模型和图片！");
        },
	    error: function(data) {
            alert(e.responseText); //alert错误信息 
        }
    });
    
    // var canvas = document.getElementById('canvas')
    // var a = document.createElement('a')
    // a.download = 'canvas'
    // a.href = canvas.toDataURL('image/png')
    // document.body.appendChild(a)
    // a.click()
    // document.body.removeChild(a)
    // // 预测obj
    // $.post("http://10.33.30.36:8899/construct",
    // function(res){
    //     if(res.code==200){
    //         useDefaultData()
    //     }
    //     else{
    //         alert('请重试')
    //     }
    // });
    // useDefaultData()
}

// 清空画布
var clean = document.getElementById('clean')
clean.onclick = function () {
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.fillStyle = 'white'
    context.fillRect(0, 0, canvas.width, canvas.height)
    var rect = canvas.getBoundingClientRect()
    var x = (0 - rect.left) / rect.width * canvas.width
    var y = (0 - rect.top) / rect.height * canvas.height
    Mouse = { x: x, y: y }
    lastMouse = { x: x, y: y }
}
