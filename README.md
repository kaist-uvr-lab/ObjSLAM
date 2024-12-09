# WISEUI-GGSG
이 프로젝트는 2019년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 프로젝트임 (No.2019-0-01270,스마트 안경을 위한 WISE AR UI/UX 플랫폼 개발)

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-01270, WISE AR UI/UX Platform Development for Smartglasses)


**How to test this project?**
1. Install the requirements.
  1) Unity 2019.4.26f1
  2) Microsoft MRTK 2.6.1 & additional requirements for Hololens2 development
  3) Photon PUN2 v.2.33 (at Unity Asset Store)

2. Build scenes for each device.
  1) Hololens2 (Build Settings = Universal Windows Platform) : NetworkLobby(0), SmartglassConnected(1)
  2) Smartphone (Build Settings = Android) : NetworkLobby(0), SmartphoneConnected(1)

3. Start the application in both devices.



**About Deprecated Version**
During project development, the overall setting has been upgraded once according to the version upgrade of the sdk. The version before the upgrade is stored in the "oldVersionArchive" branch, and the execution environment is as follows.

1. Install the requirements.
  1) Unity 2019.3.15f1
  2) Microsoft MRTK 2.4.0 & additional requirements for Hololens2 development (https://docs.microsoft.com/en-us/windows/mixed-reality/develop/unity/tutorials/mr-learning-base-01)
  3) Photon PUN2 (at Unity Asset Store - v.2.21) (https://doc.photonengine.com/zh-TW/pun/current/demos-and-tutorials/pun-basics-tutorial/intro)

2. Build scenes for each device.
  1) Hololens2 (Build Settings = Universal Windows Platform) : NetworkLobby(0), SmartglassConnected(1)
  2) Smartphone (Build Settings = Android) : NetworkLobby(0), SmartphoneConnected(1)

3. Start the application in both devices.