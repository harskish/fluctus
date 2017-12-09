#include "tracer.hpp"
#include "geom.h"

using namespace nanogui;

void Tracer::setupToolbar()
{
    Settings &s = Settings::getInstance();
    auto screen = window->getScreen();

    tools = new Window(screen, "Tools");
    tools->setLayout(new BoxLayout(Orientation::Vertical));
    tools->setPosition(Vector2i(0, 0));
    tools->setLayout(new GroupLayout());
    tools->setFixedWidth(210);
    
    // Load scene
    auto openSceneBtn = new Button(tools, "Open scene", ENTYPO_ICON_FOLDER);
    openSceneBtn->setCallback([&]() {
        init(params.width, params.height);
        paramsUpdatePending = true;
    });

    // Renderer settings
    addRendererSettings(tools);

    // Camera
    addCameraSettings(tools);

    // Environment map
    addEnvMapSettings(tools);

    // Area light
    addAreaLightSettings(tools);

    // State settings
    addStateSettings(tools);

    // Export image
    auto exportButton = new Button(tools, "Save image", ENTYPO_ICON_CAMERA);
    exportButton->setCallback([&]() {
        auto name = saveFileDialog("Save image as", "", { "*.png" });
        if (name == "") return;        
        if (!endsWith(name, ".png")) name += ".png";
        clctx->saveImage(name, params, useWavefront);
    });

    // Hide toolbar by default
    tools->setVisible(false);

    // Make visible
    screen->setVisible(true);
    screen->performLayout();
}


void Tracer::addRendererSettings(nanogui::Widget *parent)
{
    Settings &s = Settings::getInstance();

    // Renderer settings
    PopupButton *rendererBtn = new PopupButton(parent, "Renderer");
    rendererBtn->setIcon(ENTYPO_ICON_PICASA);
    rendererBtn->setBackgroundColor(Color(100, 0, 0, 25));
    Popup *rendererPopup = rendererBtn->popup();
    rendererPopup->setAnchorHeight(61);

    // Integrator
    rendererPopup->setLayout(new GroupLayout());
    new Label(rendererPopup, "Integrator", "sans-bold");
    const std::vector<std::string> intShort = { "W-PT", "M-PT", "RT" };
    const std::vector<std::string> intLong = { "Path tracer (Wavefront)", "Path tracer (Megakernel)", "Ray tracer" };
    auto integratorBox = new ComboBox(rendererPopup, intLong, intShort);
    uiMapping["INTEGRATOR_BOX"] = integratorBox;
    integratorBox->setFixedWidth(100);
    integratorBox->setCallback([&](int idx) {
        if (idx == 0)
        {
            useWavefront = true;
            window->setRenderMethod(PTWindow::RenderMethod::WAVEFRONT);
        }
        if (idx == 1)
        {
            useWavefront = false;
            window->setRenderMethod(PTWindow::RenderMethod::MEGAKERNEL);
        }
            
        paramsUpdatePending = true;
    });
    integratorBox->setSelectedIndex((useWavefront) ? 0 : 1);

    // Sampler
    new Label(rendererPopup, "Sampler settings", "sans-bold");
    auto envMapBox = new CheckBox(rendererPopup, "Environment map");
    uiMapping["ENV_MAP_TOGGLE"] = envMapBox;
    envMapBox->setChecked(params.useEnvMap);
    envMapBox->setCallback([&](bool value) {
        params.useEnvMap = value;
        paramsUpdatePending = true;
    });
    auto explSamplBox = new CheckBox(rendererPopup, "Explicit sampling");
    uiMapping["EXPL_SAMPL_TOGGLE"] = explSamplBox;
    explSamplBox->setChecked(params.sampleExpl);
    explSamplBox->setCallback([&](bool value) {
        params.sampleExpl = value;
        paramsUpdatePending = true;
    });
    auto implSamplBox = new CheckBox(rendererPopup, "Implicit sampling");
    uiMapping["IMPL_SAMPL_TOGGLE"] = implSamplBox;
    implSamplBox->setChecked(params.sampleImpl);
    implSamplBox->setCallback([&](bool value) {
        params.sampleImpl = value;
        paramsUpdatePending = true;
    });
    auto areaLightBox = new CheckBox(rendererPopup, "Area light");
    uiMapping["AREA_LIGHT_TOGGLE"] = areaLightBox;
    areaLightBox->setChecked(params.useAreaLight);
    areaLightBox->setCallback([&](bool value) {
        params.useAreaLight = value;
        paramsUpdatePending = true;
    });

    Widget *depthPanel = new Widget(rendererPopup);
    depthPanel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 5));
    IntBox<int> *depthBox = new IntBox<int>(depthPanel);
    uiMapping["MAX_BOUNCES_BOX"] = depthBox;
    depthBox->setFixedWidth(48);
    depthBox->setAlignment(TextBox::Alignment::Right);
    depthBox->setValue(std::min(99, (int)params.maxBounces));
    depthBox->setEditable(true);
    inputBoxes.push_back(depthBox);
    depthBox->setFormat("[0-9][0-9]*");
    depthBox->setSpinnable(true);
    depthBox->setMinMaxValues(0, 99);
    depthBox->setValueIncrement(1);
    depthBox->setCallback([&](int value) {
        params.maxBounces = (cl_uint)value;
        paramsUpdatePending = true;
    });
    auto depthDesc = new Label(depthPanel, "Maximum path depth");

    // Render scale
    new Label(rendererPopup, "Render scale", "sans-bold");
    Widget *renderScalePanel = new Widget(rendererPopup);
    renderScalePanel->setLayout(new BoxLayout(Orientation::Horizontal));

    Slider *slider = new Slider(renderScalePanel);
    slider->setRange(std::make_pair(0.01f, 1.5f));
    slider->setValue(s.getRenderScale());
    slider->setFixedWidth(80);

    auto box = new IntBox<int>(renderScalePanel);
    box->setEditable(true);
    inputBoxes.push_back(box);
    box->setFormat("[1-9][0-9]*");
    box->setSpinnable(true);
    box->setMinValue(1);
    box->setValueIncrement(1);
    box->setValue((int)(s.getRenderScale() * 100));
    box->setUnits("%");
    box->setFixedSize(Vector2i(65, 25));
    box->setFontSize(20);
    box->setAlignment(TextBox::Alignment::Right);

    box->setCallback([&s, slider, this](int value) {
        s.setRenderScale(value / 100.0f);
        slider->setValue(std::min(1.5f, value / 100.0f));
        resizeBuffers(params.width, params.height);
        paramsUpdatePending = true;
    });

    slider->setCallback([box](float value) {
        box->setValue((int)(value * 100));
    });

    slider->setFinalCallback([&](float value) {
        s.setRenderScale(value);
        resizeBuffers(params.width, params.height);
        paramsUpdatePending = true;
    });
}


void Tracer::addCameraSettings(Widget *parent)
{
    PopupButton *camBtn = new PopupButton(parent, "Camera");
    Popup *camPopup = camBtn->popup();
    camPopup->setAnchorHeight(61);
    camPopup->setLayout(new GroupLayout());

    // FOV
    Widget *fovWidget = new Widget(camPopup);
    fovWidget->setLayout(new BoxLayout(Orientation::Horizontal));
    Label *fovlabel = new Label(fovWidget, "FOV");
    fovlabel->setFixedWidth(40);
    auto fovSlider = new Slider(fovWidget);
    uiMapping["FOV_SLIDER"] = fovSlider;
    fovSlider->setRange(std::make_pair(0.01f, 179.0f));
    fovSlider->setValue(params.camera.fov);
    fovSlider->setFixedWidth(80);
    auto fovBox = new FloatBox<cl_float>(fovWidget);
    uiMapping["FOV_BOX"] = fovBox;
    inputBoxes.push_back(fovBox);
    fovBox->setEditable(true);
    fovBox->setMinMaxValues(0.01f, 179.0f);
    fovBox->setFormat("[0-9]*\\.?[0-9]+");
    fovBox->setCallback([fovSlider, this](cl_float val) {
        params.camera.fov = val;
        fovSlider->setValue(val);
        paramsUpdatePending = true;
    });
    fovBox->setFixedWidth(80);
    fovBox->setValue(params.camera.fov);
    fovSlider->setCallback([fovBox, this](cl_float val) {
        params.camera.fov = val;
        fovBox->setValue(val);
        paramsUpdatePending = true;
    });

    // Speed
    Widget *speedWidget = new Widget(camPopup);
    speedWidget->setLayout(new BoxLayout(Orientation::Horizontal));
    Label *speedLabel = new Label(speedWidget, "Speed");
    speedLabel->setFixedWidth(40);
    auto speedSlider = new Slider(speedWidget);
    uiMapping["CAM_SPEED_SLIDER"] = speedSlider;
    speedSlider->setRange(std::make_pair(0.1f, 100.0f));
    speedSlider->setValue(cameraSpeed);
    speedSlider->setFixedWidth(80);
    auto speedBox = new FloatBox<float>(speedWidget);
    uiMapping["CAM_SPEED_BOX"] = speedBox;
    speedBox->setFixedWidth(80);
    speedBox->setValue(cameraSpeed);
    speedBox->setEditable(true);
    inputBoxes.push_back(speedBox);
    speedBox->setFormat("[0-9]*\\.?[0-9]+");
    speedBox->setCallback([speedSlider, this](float val) {
        cameraSpeed = val;
        if (val <= 100.0f)
            speedSlider->setValue(val);
    });
    speedSlider->setCallback([speedBox, this](float val) {
        cameraSpeed = val;
        speedBox->setValue(val);
    });

    // Reset
    Button *resetButton = new Button(camPopup, "Reset");
    resetButton->setCallback([fovSlider, fovBox, speedSlider, speedBox, this]() {
        initCamera();
        fovSlider->setValue(params.camera.fov);
        fovBox->setValue(params.camera.fov);
        speedSlider->setValue(cameraSpeed);
        speedBox->setValue(cameraSpeed);
    });
}


void Tracer::addEnvMapSettings(Widget *parent)
{
    PopupButton *envBtn = new PopupButton(parent, "Environment map");
    Popup *envPopup = envBtn->popup();
    envPopup->setLayout(new GroupLayout());

    // Load
    auto envLoadBtn = new Button(envPopup, "Load", ENTYPO_ICON_FOLDER);
    envLoadBtn->setCallback([&]() {
        window->showMessage("Loading environment map");
        std::string name = openFileDialog("Select environment map", "assets/env_maps/", { "*.hdr" });
        if (name == "") return;

        Settings::getInstance().setEnvMapName(name);
        if (!envMap || envMap->getName() != name)
        {
            envMap.reset(new EnvironmentMap(name));
            scene->setEnvMap(envMap);
            initEnvMap();
            paramsUpdatePending = true;
        }
        else
        {
            std::cout << "Reusing environment map" << std::endl;
        }

        window->hideMessage();
    });

    // Strength
    Widget *envEmission = new Widget(envPopup);
    envEmission->setLayout(new BoxLayout(Orientation::Horizontal));
    new Label(envEmission, "Strength");
    Slider *esl = new Slider(envEmission);
    uiMapping["ENV_MAP_SLIDER"] = esl;
    esl->setRange(std::make_pair(0.1f, 20.0f));
    esl->setValue(1.0f);
    esl->setFixedWidth(80);

    FloatBox<cl_float> *emVal = new FloatBox<cl_float>(envEmission);
    uiMapping["ENV_MAP_BOX"] = emVal;
    emVal->setValue(1.0f);
    emVal->setEditable(true);
    inputBoxes.push_back(emVal);
    emVal->setFormat("[0-9]*\\.?[0-9]+");
    emVal->setFixedSize(Eigen::Vector2i(65, 25));
    emVal->setFontSize(20);
    emVal->setAlignment(nanogui::TextBox::Alignment::Right);

    emVal->setCallback([&](float value) {
        params.envMapStrength = value;
        paramsUpdatePending = true;
    });

    esl->setCallback([emVal, this](float value) {
        emVal->setValue(value);
        params.envMapStrength = value;
        paramsUpdatePending = true;
    });
}


void Tracer::addAreaLightSettings(Widget *parent)
{
    PopupButton *alBtn = new PopupButton(parent, "Area light");
    Popup *alPopup = alBtn->popup();
    alPopup->setLayout(new GroupLayout());

    // Size
    Widget *alSize = new Widget(alPopup);
    alSize->setLayout(new BoxLayout(Orientation::Horizontal));
    auto sl = new Label(alSize, "Size");
    sl->setFixedWidth(50);
    Slider *ssl = new Slider(alSize);
    uiMapping["AL_SIZE_SLIDER"] = ssl;
    ssl->setRange(std::make_pair(0.1f, 30.0f));
    ssl->setValue(params.areaLight.size.x);
    ssl->setFixedWidth(90);

    FloatBox<cl_float> *sVal = new FloatBox<cl_float>(alSize);
    uiMapping["AL_SIZE_BOX"] = sVal;
    sVal->setValue(params.areaLight.size.x);
    sVal->setEditable(true);
    inputBoxes.push_back(sVal);
    sVal->setFormat("[0-9]*\\.?[0-9]+");
    sVal->setFixedSize(Eigen::Vector2i(65, 25));
    sVal->setFontSize(20);
    sVal->setAlignment(nanogui::TextBox::Alignment::Right);

    sVal->setCallback([&](cl_float value) {
        params.areaLight.size.x = value;
        params.areaLight.size.y = value;
        paramsUpdatePending = true;
    });

    ssl->setCallback([sVal, this](cl_float value) {
        sVal->setValue(value);
        params.areaLight.size.x = value;
        params.areaLight.size.y = value;
        paramsUpdatePending = true;
    });


    // Intensity
    Widget *alInt = new Widget(alPopup);
    alInt->setLayout(new BoxLayout(Orientation::Horizontal));
    auto il = new Label(alInt, "Intensity");
    il->setFixedWidth(50);
    Slider *isl = new Slider(alInt);
    uiMapping["AL_INT_SLIDER"] = isl;
    isl->setRange(std::make_pair(0.1f, 100.0f));
    isl->setValue(std::sqrt(params.areaLight.E.sqnorm()));
    isl->setFixedWidth(90);

    FloatBox<cl_float> *iVal = new FloatBox<cl_float>(alInt);
    uiMapping["AL_INT_BOX"] = iVal;
    iVal->setValue(std::sqrt(params.areaLight.E.sqnorm()));
    iVal->setEditable(true);
    inputBoxes.push_back(iVal);
    iVal->setFormat("[0-9]*\\.?[0-9]+");
    iVal->setFixedSize(Eigen::Vector2i(65, 25));
    iVal->setFontSize(20);
    iVal->setAlignment(nanogui::TextBox::Alignment::Right);

    iVal->setCallback([&](cl_float value) {
        float sqnorm = params.areaLight.E.sqnorm();
        if (sqnorm == 0.0f) return;
        params.areaLight.E /= std::sqrt(sqnorm);
        params.areaLight.E *= value;
        paramsUpdatePending = true;
    });

    isl->setCallback([iVal, this](cl_float value) {
        iVal->setValue(value);
        float sqnorm = params.areaLight.E.sqnorm();
        if (sqnorm == 0.0f) return;
        params.areaLight.E /= std::sqrt(sqnorm);
        params.areaLight.E *= value;
        paramsUpdatePending = true;
    });

    // Color
    Widget *cp = new Widget(alPopup);
    cp->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
    auto cl = new Label(cp, "Color");
    cl->setFixedWidth(50);
    auto colorPicker = new nanogui::ColorPicker(cp);
    colorPicker->setFixedWidth(145);
    colorPicker->setColor(nanogui::Color(Vector3f(1.0f, 1.0f, 1.0f), 1.0f));
    colorPicker->setFinalCallback([&](const nanogui::Color &c) {
        float3 E = params.areaLight.E;
        float intensity = std::max(E.x, std::max(E.y, E.z));
        params.areaLight.E = intensity * float3(c[0], c[1], c[2]);
        paramsUpdatePending = true;
    });
}


void Tracer::addStateSettings(Widget *parent)
{
    PopupButton *stateBtn = new PopupButton(parent, "State");
    Popup *statePopup = stateBtn->popup();
    statePopup->setAnchorHeight(61);

    statePopup->setLayout(new GroupLayout());
    new Label(statePopup, "Current state", "sans-bold");
    Widget *statePanel = new Widget(statePopup);
    statePanel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 5));

    Button *resetBtn = new Button(statePanel, "Reset", ENTYPO_ICON_SQUARED_CROSS);
    resetBtn->setCallback([&] { initCamera(); paramsUpdatePending = true; });

    Button *loadStateBtn = new Button(statePanel, "Load", ENTYPO_ICON_UPLOAD);
    loadStateBtn->setCallback([&] { loadState(); paramsUpdatePending = true; });

    Button *saveStateBtn = new Button(statePanel, "Save", ENTYPO_ICON_DOWNLOAD);
    saveStateBtn->setCallback([&] { saveState(); paramsUpdatePending = true; });
}


// Update GUI sliders/boxes based on new state
void Tracer::updateGUI()
{
    auto fovBox = static_cast<FloatBox<cl_float>*>(uiMapping["FOV_BOX"]);
    auto fovSlider = static_cast<Slider*>(uiMapping["FOV_SLIDER"]);
    fovBox->setValue(params.camera.fov);
    fovSlider->setValue(params.camera.fov);
    
    auto envBox = static_cast<FloatBox<cl_float>*>(uiMapping["ENV_MAP_BOX"]);
    auto envSlider = static_cast<Slider*>(uiMapping["ENV_MAP_SLIDER"]);
    envSlider->setValue(params.envMapStrength);
    envBox->setValue(params.envMapStrength);

    auto alSizeBox = static_cast<FloatBox<cl_float>*>(uiMapping["AL_SIZE_BOX"]);
    auto alSizeSlider = static_cast<Slider*>(uiMapping["AL_SIZE_SLIDER"]);
    alSizeBox->setValue(params.areaLight.size.x);
    alSizeSlider->setValue(params.areaLight.size.x);

    auto alIntBox = static_cast<FloatBox<cl_float>*>(uiMapping["AL_INT_BOX"]);
    auto alIntSlider = static_cast<Slider*>(uiMapping["AL_INT_SLIDER"]);
    alIntBox->setValue(std::sqrt(params.areaLight.E.sqnorm()));
    alIntSlider->setValue(std::sqrt(params.areaLight.E.sqnorm()));

    auto envMapToggle = static_cast<CheckBox*>(uiMapping["ENV_MAP_TOGGLE"]);
    auto explSampleToggle = static_cast<CheckBox*>(uiMapping["EXPL_SAMPL_TOGGLE"]);
    auto implSampleToggle = static_cast<CheckBox*>(uiMapping["IMPL_SAMPL_TOGGLE"]);
    auto areaLightToggle = static_cast<CheckBox*>(uiMapping["AREA_LIGHT_TOGGLE"]);
    envMapToggle->setChecked(params.useEnvMap);
    explSampleToggle->setChecked(params.sampleExpl);
    implSampleToggle->setChecked(params.sampleImpl);
    areaLightToggle->setChecked(params.useAreaLight);
    
    auto maxBouncesBox = static_cast<IntBox<int>*>(uiMapping["MAX_BOUNCES_BOX"]);
    maxBouncesBox->setValue(params.maxBounces);

    auto camSpeedSlider = static_cast<Slider*>(uiMapping["CAM_SPEED_SLIDER"]);
    auto camSpeedBox = static_cast<FloatBox<float>*>(uiMapping["CAM_SPEED_BOX"]);
    camSpeedSlider->setValue(cameraSpeed);
    camSpeedBox->setValue(cameraSpeed);

    auto integratorBox = static_cast<ComboBox*>(uiMapping["INTEGRATOR_BOX"]);
    integratorBox->setSelectedIndex((useWavefront) ? 0 : 1);
}


void Tracer::toggleGUI()
{
    tools->setVisible(!tools->visible());
    window->getScreen()->performLayout();
}


// Do not poll when text input is in progress
bool Tracer::shouldSkipPoll()
{
    for (TextBox* b : inputBoxes)
    {
        if (b->focused()) return true;
    }    

    return false;
}