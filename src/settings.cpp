#include <fstream>
#include "settings.hpp"

using json = nlohmann::json;

Settings::Settings()
{
    init();
    load();
}

void Settings::init()
{
    platformName = "";
    deviceName = "";
    envMapName = "";
    renderScale = 1.0f;
    windowWidth = 640;
    windowHeight = 480;
    wfBufferSize = 1 << 20; // appropriate for dedicated GPU
    clUseBitstack = false;
    clUseSoA = true;
}

inline bool contains(json j, std::string value)
{
    return j.find(value) != j.end();
}

void Settings::load()
{
    std::ifstream i("settings.json");

    if(!i.good())
    {
        std::cout << "Settings file not found!" << std::endl;
        return;
    }

    json j;
    i >> j;

    if(!contains(j, "release") || !contains(j, "debug"))
    {
        std::cout << "Settings file must contain the objects \"release\" and \"debug\"" << std::endl;
        return;
    }

    // Read release settings first
    import(j["release"]);

#ifdef _DEBUG
    // Override with debug settings in debug mode
    import(j["debug"]);
#endif
}

void Settings::import(json j)
{
    if (contains(j, "platformName")) this->platformName = j["platformName"].get<std::string>();
    if (contains(j, "deviceName")) this->deviceName = j["deviceName"].get<std::string>();
    if (contains(j, "envMap")) this->envMapName = j["envMap"].get<std::string>();
    if (contains(j, "renderScale")) this->renderScale = j["renderScale"].get<float>();
    if (contains(j, "windowWidth")) this->windowWidth = j["windowWidth"].get<int>();
    if (contains(j, "windowHeight")) this->windowHeight = j["windowHeight"].get<int>();
    if (contains(j, "clUseBitstack")) this->clUseBitstack = j["clUseBitstack"].get<bool>();
    if (contains(j, "clUseSoA")) this->clUseSoA = j["clUseSoA"].get<bool>();
    if (contains(j, "wfBufferSize")) this->wfBufferSize = j["wfBufferSize"].get<unsigned int>();

    // Map of numbers 1-5 to scenes (shortcuts)
    if (contains(j, "shortcuts"))
    {
        json map = j["shortcuts"];
        for (unsigned int i = 1; i < 6; i++)
        {
            std::string numeral = std::to_string(i);
            if (contains(map, numeral)) this->shortcuts[i] = map[numeral].get<std::string>();
        }
    }
}