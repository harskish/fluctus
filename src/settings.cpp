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
    if(contains(j, "platformName")) this->platformName = j["platformName"];
    if(contains(j, "deviceName")) this->deviceName = j["deviceName"];
}