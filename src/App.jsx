import * as mobilenet from '@tensorflow-models/mobilenet'
import * as knnClassifier from '@tensorflow-models/knn-classifier'
import { Howl } from 'howler'
import { useEffect, useRef, useState } from 'react'

var sound = new Howl({
  src: ['/src/assets/hey_sondn.mp3']
});


function App() {
  const TOUCH = 'touch'
  const NOT_TOUCH = 'not touch'
  const TRAINING_TIMES = 50
  const CONFIDENCE_THRESHOLD = 0.6
  const videoRef = useRef()
  const mobilenetModel = useRef()
  const classifier = useRef()
  const canPlaySound = useRef(true)
  const [isTouching, setIsTouching] = useState(false)

  async function init() {
    console.log('init ...')
    await setupCamera()
    console.log('camera running ...')
  }

  function setupCamera() {
    return new Promise((resolve, reject) => {
      navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mzGetUserMedia
      if (navigator.getUserMedia) {
        navigator.getUserMedia(
          { video: { width: 480, height: 360 } },
          // { video: true }, 
          stream => {
            const videoStream = videoRef.current
            videoStream.srcObject = stream
            videoStream.onloadedmetadata = (e) => {
              videoStream.play();
              resolve()
            }
          },
          error => reject(error)
        )
      }
      else {
        reject('Camera not found!')
      }
    })
  }

  async function batchTraining(label) {
    console.log(`[${label}] Capturing faces for training ...`)
    for (let i = 0; i < TRAINING_TIMES; i++) {
      console.log(`Process: ${((i + 1) / TRAINING_TIMES * 100)}%`)
      await individualTraining(label)
    }
    alert('Training Done!')
  }

  function individualTraining(label) {
    return new Promise(async resolve => {
      const videoFrameEmbedding = mobilenetModel.current.infer(videoRef.current, true)
      classifier.current.addExample(videoFrameEmbedding, label)
      await sleep(100)
      resolve()
    })
  }

  function sleep(ms) {
    return new Promise(resolve => {
      setTimeout(resolve, ms)
    })
  }

  async function run() {
    const currentVideoFrameEmbedding = mobilenetModel.current.infer(videoRef.current, true)
    const result = await classifier.current.predictClass(currentVideoFrameEmbedding)
    // console.log(result.label, result.confidences)
    if (result.label === TOUCH && result.confidences[TOUCH] >= CONFIDENCE_THRESHOLD) {
      canPlaySound.current = false
      setIsTouching(true)
      sound.play()
    }
    else {
      setIsTouching(false)
    }
    await sleep(500)
    run()
  }

  useEffect(() => {
    init()
    async function loadModels() {
      // Load mobilenet model.
      mobilenetModel.current = await mobilenet.load();
      // create knn classifier
      classifier.current = knnClassifier.create()
      console.log('load models successfully')
    }
    sound.on('end', function() {
      canPlaySound.current = true
    })
    loadModels()
    return () => {

    }
  }, [])

  return (
    <>
      <div className={"flex flex-col justify-center items-center h-screen " + (isTouching ? "bg-red-600" : "")}>
        <video ref={videoRef} className="w-[480px] h-[360px] bg-gray-800"></video>
        <div className="mt-4">
          <button className="px-4 py-1 text-xl bg-green-400 border mr-2" onClick={() => batchTraining(NOT_TOUCH)}>Train 1</button>
          <button className="px-4 py-1 text-xl bg-green-400 border mr-2" onClick={() => batchTraining(TOUCH)}>Train 2</button>
          <button className="px-4 py-1 text-xl bg-green-400 border mr-2" onClick={() => run()}>Run</button>
        </div>
      </div>
    </>
  )
}

export default App
