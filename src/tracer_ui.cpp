#include "tracer.hpp"
#include "clcontext.hpp"
#include "settings.hpp"
#include "window.hpp"
#include "geom.h"
#include "utils.h"
#include <functional>

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

    // Tonemapping
    addTonemapSettings(tools);

    // Optix denoiser
    addDenoiserSettings(tools);

    // Environment map
    addEnvMapSettings(tools);

    // Area light
    addAreaLightSettings(tools);

    // State settings
    addStateSettings(tools);

    // Run benchmark
    auto benchmarkButton = new Button(tools, "Benchmark", ENTYPO_ICON_GAUGE);
    benchmarkButton->setCallback([&]() {
        runBenchmark();
    });
    
    // Export image
    auto exportButton = new Button(tools, "Save image", ENTYPO_ICON_CAMERA);
    exportButton->setCallback([&]() {
        auto name = saveFileDialog("Save image as", "", { "*.png", "*.hdr", "*.bmp" });
        if (name == "") return;        
        if (name.find('.') == std::string::npos) name += ".png";
#ifdef WITH_OPTIX
        if (useDenoiser)
            denoiser.denoise();
#endif
        clctx->saveImage(name, params);
    });

    // Hide toolbar by default
    tools->setVisible(false);

    // Make visible
    screen->setVisible(true);
    screen->performLayout();
}


struct FloatWidget
{
    Widget* root;
    Label* label;
    Slider* slider;
    FloatBox<cl_float>* box;
};

FloatWidget* Tracer::addFloatWidget(Popup* parent, std::string title, std::string key, float vMin, float vMax, std::function<void(float)> updateFunc)
{
    FloatWidget* w = new FloatWidget();
    w->root = new Widget(parent);
    w->root->setLayout(new BoxLayout(Orientation::Horizontal));
    w->label = new Label(w->root, title);
    w->label->setFixedWidth(60);
    w->slider = new Slider(w->root);
    uiMapping[key + "_SLIDER"] = w->slider;
    w->slider->setRange(std::make_pair(vMin, vMax));
    w->slider->setFixedWidth(100);
    w->box = new FloatBox<cl_float>(w->root);
    uiMapping[key + "_BOX"] = w->box;
    inputBoxes.push_back(w->box);
    w->box->setEditable(true);
    w->box->setMinMaxValues(vMin, vMax);
    w->box->setFormat("[0-9]*\\.?[0-9]+");
    w->box->setFixedWidth(80);
    w->box->setValue(params.ppParams.exposure);

    w->slider->setCallback([w, updateFunc, this](cl_float val) {
        w->box->setValue(val);
        updateFunc(val);
    });

    w->box->setCallback([w, updateFunc, this](cl_float val) {
        w->slider->setValue(val);
        updateFunc(val);
    });

    return w;
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
    const std::vector<std::string> intShort = { "W-PT", "M-PT" };
    const std::vector<std::string> intLong = { "Path tracer (Wavefront)", "Path tracer (Microkernel)" };
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
            window->setRenderMethod(PTWindow::RenderMethod::MICROKERNEL);
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
    auto rrBox = new CheckBox(rendererPopup, "Russian roulette");
    uiMapping["RR_TOGGLE"] = rrBox;
    rrBox->setChecked(params.useRoulette);
    rrBox->setCallback([&](bool value) {
        params.useRoulette = value;
        paramsUpdatePending = true;
    });
    auto matQueueBox = new CheckBox(rendererPopup, "Separate material queues (WF)");
    uiMapping["WF_QUEUE_TOGGLE"] = matQueueBox;
    matQueueBox->setChecked(params.wfSeparateQueues);
    matQueueBox->setCallback([&](bool value) {
        params.wfSeparateQueues = value;
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
    uiMapping["RENDER_SCALE_SLIDER"] = slider;

    auto box = new IntBox<int>(renderScalePanel);
    uiMapping["RENDER_SCALE_BOX"] = box;
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
    //camPopup->setAnchorHeight(61);
    camPopup->setLayout(new GroupLayout());

    // FOV
    FloatWidget* fovWidget = addFloatWidget(camPopup, "FOV", "FOV", 0.01f, 120.0f, [this](float val) {
        params.camera.fov = (cl_float)val;
        paramsUpdatePending = true;
    });

    // Speed
    FloatWidget* speedWidget = addFloatWidget(camPopup, "Speed", "CAM_SPEED", 0.1f, 100.0f, [this](float val) {
        cameraSpeed = val;
        paramsUpdatePending = true;
    });

    // Aperture size
    FloatWidget* apertureWidget = addFloatWidget(camPopup, "Aperture", "CAM_APERTURE", 0.0f, 0.003f, [this](float val) {
        params.camera.apertureSize = (cl_float)val;
        paramsUpdatePending = true;
    });

    // Help message
    new Label(camPopup, "Right click to set focal distance");

    // Reset
    Button *resetButton = new Button(camPopup, "Reset");
    resetButton->setCallback([fovWidget, speedWidget, apertureWidget, this]() {
        initCamera();
        fovWidget->slider->setValue(params.camera.fov);
        fovWidget->box->setValue(params.camera.fov);
        speedWidget->slider->setValue(cameraSpeed);
        speedWidget->box->setValue(cameraSpeed);
        apertureWidget->slider->setValue(params.camera.apertureSize);
        apertureWidget->box->setValue(params.camera.apertureSize);
    });
}


void Tracer::addTonemapSettings(Widget *parent)
{
    PopupButton *tmBtn = new PopupButton(parent, "Tonemapping");
    Popup *tmPopup = tmBtn->popup();
    tmPopup->setLayout(new GroupLayout());

    // Exposure
    FloatWidget* expWidget = addFloatWidget(tmPopup, "Exposure", "EXPOSURE", 0.1f, 5.0f, [this](float val) {
        params.ppParams.exposure = val;
        clctx->updateParams(params);
    });

    // Tonemapping operator
    Widget *opWidget = new Widget(tmPopup);
    opWidget->setLayout(new BoxLayout(Orientation::Horizontal));
    auto opLabel = new Label(opWidget, "Operator");
    opLabel->setFixedWidth(60);
    const std::vector<std::string> desc = { "Linear", "Reinhard", "Uncharted 2" };
    auto opBox = new ComboBox(opWidget, desc);
    uiMapping["TONEMAP_OP_BOX"] = opBox;
    opBox->setFixedWidth(172);
    opBox->setCallback([&](int idx) {
        params.ppParams.tmOperator = idx;
        clctx->updateParams(params);
    });
    opBox->setSelectedIndex(2);

    // Reset
    Button *resetButton = new Button(tmPopup, "Reset");
    resetButton->setCallback([expWidget, opBox, this]() {
        initPostProcessing();
        expWidget->slider->setValue(params.ppParams.exposure);
        expWidget->box->setValue(params.ppParams.exposure);
        opBox->setSelectedIndex(2);
    });
}


void Tracer::addDenoiserSettings(Widget *parent)
{
    (void)parent;
#ifdef WITH_OPTIX
    PopupButton *denoiserButton = new PopupButton(parent, "Denoiser");
    Popup *denoiserPopup = denoiserButton->popup();
    denoiserPopup->setLayout(new GroupLayout());

    auto denoiserBox = new CheckBox(denoiserPopup, "Enable");
    uiMapping["DENOISE_TOGGLE"] = denoiserBox;
    denoiserBox->setChecked(useDenoiser);
    denoiserBox->setCallback([&](bool value) {
        useDenoiser = value;
        paramsUpdatePending = true; // rebuild kernels
    });

    FloatWidget* mixWidget = addFloatWidget(denoiserPopup, "Strength", "DENOISER_BLEND", 0.0f, 1.0f, [this](float val) {
        denoiser.setBlend(1.0f - val);
        denoiserStrength = val;
    });
#endif
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
    FloatWidget* envEmissionWidget = addFloatWidget(envPopup, "Strength", "ENV_MAP", 0.1f, 10.0f, [this](float val) {
        params.envMapStrength = (cl_float)val;
        paramsUpdatePending = true;
    });
}


void Tracer::addAreaLightSettings(Widget *parent)
{
    PopupButton *alBtn = new PopupButton(parent, "Area light");
    Popup *alPopup = alBtn->popup();
    alPopup->setLayout(new GroupLayout());

    // Size
    FloatWidget* sizeWidget = addFloatWidget(alPopup, "Size", "AL_SIZE", 0.1f, 30.0f, [this](float val) {
        params.areaLight.size.x = (cl_float)val;
        params.areaLight.size.y = (cl_float)val;
        paramsUpdatePending = true;
    });

    // Intensity
    FloatWidget* intensityWidget = addFloatWidget(alPopup, "Intensity", "AL_INT", 0.1f, 100.0f, [this](float val) {
        float sqnorm = params.areaLight.E.sqnorm();
        if (sqnorm == 0.0f) return;
        params.areaLight.E /= std::sqrt(sqnorm);
        params.areaLight.E *= val;
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
    auto rrToggle = static_cast<CheckBox*>(uiMapping["RR_TOGGLE"]);
    auto matQueueToggle = static_cast<CheckBox*>(uiMapping["WF_QUEUE_TOGGLE"]);
    envMapToggle->setChecked(params.useEnvMap);
    explSampleToggle->setChecked(params.sampleExpl);
    implSampleToggle->setChecked(params.sampleImpl);
    areaLightToggle->setChecked(params.useAreaLight);
    rrToggle->setChecked(params.useRoulette);
    matQueueToggle->setChecked(params.wfSeparateQueues);

#ifdef WITH_OPTIX    
    auto denoiseToggle = static_cast<CheckBox*>(uiMapping["DENOISE_TOGGLE"]);
    denoiseToggle->setChecked(useDenoiser);

    auto strengthSlider = static_cast<Slider*>(uiMapping["DENOISER_BLEND_SLIDER"]);
    auto strengthBox = static_cast<FloatBox<float>*>(uiMapping["DENOISER_BLEND_BOX"]);
    strengthSlider->setValue(denoiserStrength);
    strengthBox->setValue(denoiserStrength);
#endif
    
    auto maxBouncesBox = static_cast<IntBox<int>*>(uiMapping["MAX_BOUNCES_BOX"]);
    maxBouncesBox->setValue(params.maxBounces);

    auto camSpeedSlider = static_cast<Slider*>(uiMapping["CAM_SPEED_SLIDER"]);
    auto camSpeedBox = static_cast<FloatBox<float>*>(uiMapping["CAM_SPEED_BOX"]);
    auto apertureBox = static_cast<FloatBox<float>*>(uiMapping["CAM_APERTURE_BOX"]);
    auto apertureSlider = static_cast<Slider*>(uiMapping["CAM_APERTURE_SLIDER"]);
    camSpeedSlider->setValue(cameraSpeed);
    camSpeedBox->setValue(cameraSpeed);
    apertureBox->setValue(params.camera.apertureSize);
    apertureSlider->setValue(params.camera.apertureSize);

    auto expBox = static_cast<FloatBox<float>*>(uiMapping["EXPOSURE_BOX"]);
    auto expSlider = static_cast<Slider*>(uiMapping["EXPOSURE_SLIDER"]);
    auto opBox = static_cast<ComboBox*>(uiMapping["TONEMAP_OP_BOX"]);
    expBox->setValue(params.ppParams.exposure);
    expSlider->setValue(params.ppParams.exposure);
    opBox->setSelectedIndex(params.ppParams.tmOperator);

    auto integratorBox = static_cast<ComboBox*>(uiMapping["INTEGRATOR_BOX"]);
    integratorBox->setSelectedIndex((useWavefront) ? 0 : 1);

    auto scaleSlider = static_cast<Slider*>(uiMapping["RENDER_SCALE_SLIDER"]);    
    scaleSlider->setValue(Settings::getInstance().getRenderScale());

    auto scaleBox = static_cast<IntBox<int>*>(uiMapping["RENDER_SCALE_BOX"]);
    scaleBox->setValue((int)(Settings::getInstance().getRenderScale() * 100));
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
