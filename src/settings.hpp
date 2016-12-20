#pragma once

#include <string>
#include "json.hpp"

class Settings
{
public:
    // Singleton pattern
    static Settings &getInstance() {
        static Settings instance;
        return instance;
    }
    Settings(Settings const&) = delete;
    void operator=(Settings const&) = delete;

    std::string getPlatformName() { return platformName; }
    std::string getDeviceName() { return deviceName; }

private:
    Settings();
    void init();
    void load();
    void import(nlohmann::json j);

    std::string platformName = "";
    std::string deviceName = "";
};
